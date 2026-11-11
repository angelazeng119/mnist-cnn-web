# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


# def concrete_cost():
#     # 获取用户输入
#     cement_cost_per_liter = float(input("please enter the cost of cement per liter: ："))
#     sand_cost_per_liter = float(input("please enter the cost of sand per liter: ："))
#     total_volume = float(input("please enter the total volume: ："))
#
#     # 配比 水泥:沙子 = 2 : 2.5
#     ratio_cement = 2
#     ratio_sand = 2.5
#     total_ratio = ratio_cement + ratio_sand
#
#     # 计算各原料体积
#     cement_volume = total_volume * ratio_cement / total_ratio
#     sand_volume = total_volume * ratio_sand / total_ratio
#
#     # 计算总成本
#     total_cost = cement_volume * cement_cost_per_liter + sand_volume * sand_cost_per_liter
#
#     # 输出结果
#     print(f"\n水泥用量：{cement_volume:.2f} 升")
#     print(f"沙子用量：{sand_volume:.2f} 升")
#     print(f"制造该预拌混凝土总成本：{total_cost:.2f}")
#
# # 调用函数
# concrete_cost()

def concrete_cost():
    cement_cost = float(input("please enter the cement cost: ："))
    sand_cost = float(input("请输入每升沙子的成本："))
    volume = float(input("请输入要制造的预拌混凝土体积（升）："))

    # 水泥与沙子的比例为 2 : 2.5
    total_parts = 2 + 2.5
    cement_volume = volume * 2 / total_parts
    sand_volume = volume * 2.5 / total_parts

    total_cost = cement_volume * cement_cost + sand_volume * sand_cost
    print(f"制造 {volume:g} 升预拌混凝土的成本为：{total_cost:.2f}")


concrete_cost()


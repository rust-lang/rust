# ViewWay Rust 开发待办

> 仓库: ViewWay/rust (Rust 编译器 Fork)
> Issue 来源: rust-lang/rust
> 更新时间: 2026-03-14 21:30

## 📊 Issue 统计

| 类型 | 数量 |
|------|------|
| **总 Open Issues** | **12,217** |
| Beta 回归 | 14 |
| ICE (编译器崩溃) | 118 |
| 编译器 Bug | 284 |

---

## 🔴 Phase 1: Beta 回归修复 (14 个)

### P0 - 高影响 + 已定位根因

| # | 标题 | 影响 | 根因 PR | 状态 |
|---|------|------|---------|------|
| [#153731](https://github.com/rust-lang/rust/issues/153731) | dyn compatible: 含关联常量的 trait 不能用于 dyn | **254 个 crate** | #150843 | ✅ 已修复 |
| [#153765](https://github.com/rust-lang/rust/issues/153765) | const Deref: trait bound not satisfied | 1 crate | #149375 | 🔴 待修复 |

### P1 - 有 MCVE + 已定位

| # | 标题 | 问题 | 状态 |
|---|------|------|------|
| [#153816](https://github.com/rust-lang/rust/issues/153816) | Fn trait closure 推断错误 | 闭包被推断为 FnOnce 而非 Fn | 🔴 待修复 |
| [#153850](https://github.com/rust-lang/rust/issues/153850) | temporary value dropped | 临时值生命周期问题 | 🔴 待修复 |
| [#153851](https://github.com/rust-lang/rust/issues/153851) | imports need explicit naming | 宏中 use 语句需显式命名 | 🔴 待修复 |

### P2 - LLVM 相关

| # | 标题 | 问题 | 状态 |
|---|------|------|------|
| [#153397](https://github.com/rust-lang/rust/issues/153397) | LLVM ERROR Apple M5 | LLVM 22 target-features 空字符串 | 🟡 需 LLVM 修复 |
| [#153852](https://github.com/rust-lang/rust/issues/153852) | LLVM displacement error | 内联汇编位移值超出范围 | 🟡 需 LLVM 修复 |

### P3 - 需要更多信息

| # | 标题 | 问题 | 状态 |
|---|------|------|------|
| [#153854](https://github.com/rust-lang/rust/issues/153854) | parser stack overflow | 需要复现 | 🟡 S-needs-repro |
| [#153849](https://github.com/rust-lang/rust/issues/153849) | no method named accept | 需要复现和二分 | 🟡 E-needs-mcve |

### ⚪ 不需要修复 (用户代码错误)

| # | 标题 | 说明 |
|---|------|------|
| #153764 | malformed feature attribute | 用户误用 `#[feature("name")]` 格式 |

---

## 🟠 Phase 2: ICE 修复 (118 个)

### 优先处理 (有复现 + 新增)

| # | 标题 | 标签 | 状态 |
|---|------|------|------|
| [#153861](https://github.com/rust-lang/rust/issues/153861) | ICE: trait bound not satisfied | const_trait_impl | 🔴 待修复 |
| [#153860](https://github.com/rust-lang/rust/issues/153860) | ICE: optimized_mir for constants | const_closures | 🔴 待修复 |
| [#153855](https://github.com/rust-lang/rust/issues/153855) | ICE: index out of bounds | unboxed_closures | 🔴 待修复 |
| [#153848](https://github.com/rust-lang/rust/issues/153848) | ICE: DefId is not a module | - | 🔴 待修复 |
| [#153842](https://github.com/rust-lang/rust/issues/153842) | ICE: inconsistent import resolution | - | 🔴 待修复 |
| [#153837](https://github.com/rust-lang/rust/issues/153837) | ICE: empty class stack parent | rustdoc | 🔴 待修复 |
| [#153833](https://github.com/rust-lang/rust/issues/153833) | ICE: broken MIR in AsyncDropGlue | async_drop | 🔴 待修复 |
| [#153831](https://github.com/rust-lang/rust/issues/153831) | ICE: Unevaluated ty::Const | min_generic_const_args | 🔴 待修复 |

---

## 📌 进行中

- 无

## ✅ 已完成

- [x] 克隆仓库 (2026-03-14)
- [x] 统计 issue 数量 (2026-03-14)
- [x] 分析 Beta 回归问题 (2026-03-14)

---

## 🎯 推荐修复顺序

1. **#153731** - dyn compatible (影响最大，254 个 crate)
2. **#153765** - const Deref (已定位根因)
3. **#153816** - Fn trait closure (有 MCVE)
4. **#153850** - temporary value dropped (有 MCVE)

---

**下一步**: 选择一个具体 issue 开始修复

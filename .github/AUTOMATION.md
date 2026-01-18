# GitHub Actions 自动化工作流

本仓库包含三个自动化工作流，用于简化向 Rust 上游仓库贡献的流程。

## 📋 目录

- [工作流概览](#工作流概览)
- [配置步骤](#配置步骤)
- [使用指南](#使用指南)
- [分支命名约定](#分支命名约定)
- [故障排除](#故障排除)

## 🔄 工作流概览

### 1. Sync Upstream (`sync-upstream.yml`)

**功能**: 自动同步上游仓库 (rust-lang/rust) 到 fork 仓库

- 🕐 **运行频率**: 每天 UTC 00:00
- 🎯 **目标分支**: `main`
- ⚡ **手动触发**: 支持在 Actions 页面手动触发

**作用**:
- 保持 fork 的 `main` 分支与上游同步
- 自动合并上游的最新更改
- 避免创建 PR 时出现冲突

### 2. Create PR to Upstream (`create-pr.yml`)

**功能**: 自动为修复分支创建 Pull Request 到上游

- 🚀 **触发条件**: 推送符合命名约定的分支
- 🏷️ **自动标签**: 自动添加 `T-compiler` 标签
- 🔍 **智能检测**: 避免重复创建 PR

### 3. Cleanup Merged Branches (`cleanup-branches.yml`)

**功能**: 自动删除已合并到上游的分支

- 🕐 **运行频率**: 每小时
- 🧹 **清理对象**: 已合并到上游 main 的分支
- 🛡️ **安全保护**: 不会删除 `main` 分支

## 🔧 配置步骤

### 步骤 1: 创建 Personal Access Token (PAT)

1. 访问 https://github.com/settings/tokens
2. 点击 "Generate new token (classic)"
3. 配置 token 权限:
   ```
   ✅ repo (Full control of private repositories)
   ✅ public_repo (Access public repositories)
   ```
4. 生成 token 并复制

### 步骤 2: 配置仓库 Secrets

1. 进入你的 fork 仓库页面
2. 点击 Settings > Secrets and variables > Actions
3. 添加以下 secrets:

| Name | Value | Required |
|------|-------|----------|
| `UPSTREAM_REPO` | `rust-lang/rust` | ✅ Yes |
| `PAT` | 你的 GitHub Token | ✅ Yes |

### 步骤 3: 启用 Workflows

1. 进入 Actions 页面
2. 确认三个工作流都已启用
3. 可以手动测试 "Sync Upstream" 工作流

## 📖 使用指南

### 标准贡献流程

```bash
# 1. 确保在 main 分支且与上游同步
git checkout main
git pull origin main

# 2. 创建新的修复分支 (使用命名约定)
git checkout -b fix/descriptive-name

# 3. 进行修改并提交
git add .
git commit -m "Fix: 描述你的修改"

# 4. 推送到 fork 仓库
git push -u origin fix/descriptive-name

# 5. 等待 "Create PR" 工作流自动创建 PR
#    或手动创建: gh pr create --repo rust-lang/rust
```

### 手动触发同步

如果需要立即同步上游:

1. 进入 GitHub Actions 页面
2. 选择 "Sync Upstream"
3. 点击 "Run workflow"
4. 选择 "强制同步" 选项（可选）

## 🏷️ 分支命名约定

使用以下前缀以触发自动 PR 创建:

| 前缀 | 用途 | 示例 |
|------|------|------|
| `fix/` | Bug 修复 | `fix/type-inference-error` |
| `feat/` | 新功能 | `feat/add-new-lint` |
| `refactor/` | 代码重构 | `refactor/improve-caching` |
| `impl/` | 实现功能 | `impl/async-fn` |
| `chore/` | 杂项 | `chore/update-tests` |

## 🐛 故障排除

### PR 没有自动创建

**检查**:
1. 分支名称是否遵循命名约定
2. Secrets 是否正确配置
3. 查看 Actions 运行日志

### 同步失败

**检查**:
1. PAT 是否有 `repo` 权限
2. `UPSTREAM_REPO` 是否设置为 `rust-lang/rust`
3. 检查 Actions 日志中的错误信息

### 分支没有自动删除

**原因**:
- PR 尚未合并到上游
- 分支不是从上游 main 分支创建的

**解决**:
- 等待 PR 合并
- 手动删除已不再需要的分支

## 📚 相关资源

- [Rust 贡献指南](https://rustc-dev-guide.rust-lang.org/contributing.html)
- [Rust 编译器开发指南](https://rustc-dev-guide.rust-lang.org/)
- [GitHub Actions 文档](https://docs.github.com/en/actions)

## 📝 维护

这些工作流存储在 `.github/workflows/` 目录下:

- `sync-upstream.yml` - 同步上游
- `create-pr.yml` - 创建 PR
- `cleanup-branches.yml` - 清理分支

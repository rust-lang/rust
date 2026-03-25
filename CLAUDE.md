# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a custom fork of the Rust compiler (rustc) with experimental MLIR codegen backend and Triton integration. It extends the standard Rust compiler with:

- **rustc_codegen_mlir**: A new MLIR-based codegen backend for Rust (`compiler/rustc_codegen_mlir/`)
- **Triton integration**: OpenAI's Triton compiler embedded in `src/triton/`
- **LLVM 22.0**: Uses LLVM version 22.0 with custom patches
- **Base version**: Built on top of Rust 1.93.0

This repository requires understanding of both the Rust compiler internals and MLIR/Triton ecosystem.

## Build System: x.py

The build system uses `x.py` (Python-based bootstrap system). All commands should be run via `./x.py` or the platform-specific wrapper (`x` on Unix, `x.ps1` on Windows).

### Common Build Commands

```bash
# Initial setup (interactive, configures profile, LSP, git hooks)
./x.py setup

# Build the compiler and standard library
./x.py build

# Build specific components
./x.py build compiler/rustc_codegen_mlir  # Build MLIR codegen backend
./x.py build library/std                   # Build standard library
./x.py build compiler                      # Build compiler only

# Check compilation without producing artifacts (faster)
./x.py check

# Build documentation
./x.py doc

# Run the full test suite
./x.py test

# Run specific test suites
./x.py test tests/ui                       # UI tests
./x.py test compiler/rustc_codegen_mlir    # MLIR backend tests
./x.py test library/std                    # Standard library tests

# Run a single test file
./x.py test tests/ui/some_test.rs

# Run tests with specific options
./x.py test --stage 1                      # Test with stage 1 compiler
./x.py test --keep-stage 1                 # Reuse stage 1 artifacts

# Format code
./x.py fmt

# Run clippy
./x.py clippy

# Clean build artifacts
./x.py clean
```

### Build Stages

The Rust build system uses a multi-stage bootstrap process:

- **Stage 0**: Downloaded prebuilt compiler (bootstrap compiler)
- **Stage 1**: Compiler built using stage 0, links against stage 0 std
- **Stage 2**: Compiler built using stage 1, links against stage 1 std (production-ready)

Most development uses `--stage 1` for faster iteration. Only use stage 2 when testing final compiler behavior.

### Configuration

Build configuration is in `bootstrap.toml` (see `bootstrap.example.toml` for all options):

```bash
# Generate configuration interactively
./x.py setup

# Or manually create bootstrap.toml
cp bootstrap.example.toml bootstrap.toml
# Edit bootstrap.toml as needed
```

Key configuration options:
- `llvm.download-ci-llvm = true` - Download prebuilt LLVM (recommended for faster builds)
- `rust.debug-assertions = true` - Enable debug assertions in the compiler
- `build.extended = true` - Build additional tools like Cargo

## Repository Structure

### Compiler Architecture

```
compiler/
├── rustc/                    # Main driver, delegates to backend
├── rustc_codegen_llvm/       # Default LLVM backend
├── rustc_codegen_mlir/       # MLIR backend (custom addition)
├── rustc_codegen_ssa/        # Shared codegen infrastructure
├── rustc_codegen_gcc/        # GCC backend (alternative)
├── rustc_codegen_cranelift/  # Cranelift backend (alternative)
├── rustc_middle/             # Compiler middle layer (HIR → MIR)
├── rustc_mir_transform/      # MIR optimization passes
├── rustc_borrowck/           # Borrow checker
├── rustc_hir/                # High-level IR (HIR)
├── rustc_ast/                # Abstract Syntax Tree
└── rustc_llvm/               # LLVM bindings
```

### Key Directories

- `compiler/` - All compiler crates (~80 crates)
- `library/` - Standard library (std, core, alloc, etc.)
- `src/bootstrap/` - Bootstrap build system
- `src/triton/` - Triton compiler integration (submodule)
- `src/llvm-project/` - LLVM 22.0 source
- `tests/` - Comprehensive test suites
- `src/tools/` - Additional tooling (cargo, rustfmt, clippy, etc.)

### Codegen Backend Flow

When adding or modifying the MLIR backend:

1. **AST** (rustc_ast) - Parse source code
2. **HIR** (rustc_hir) - Lower to High-level IR
3. **MIR** (rustc_middle) - Lower to Mid-level IR (control flow, borrow checking)
4. **Codegen Backend** - Lower MIR to target representation:
   - LLVM backend → LLVM IR → native code
   - MLIR backend → MLIR dialects → (various targets)
   - GCC backend → GIMPLE → native code

The MLIR backend in `compiler/rustc_codegen_mlir/` implements the `CodegenBackend` trait to integrate with rustc's compilation pipeline.

## Dependencies

### System Requirements

- Python 3 (or 2.7)
- git
- C compiler (gcc/clang for Unix, MSVC for Windows)
- curl (Unix only)
- pkg-config (Linux)
- OpenSSL development libraries (for Cargo)

### LLVM Build Requirements (if building from source)

- g++/clang++ (version per LLVM docs)
- ninja or GNU make 3.81+
- cmake (version per LLVM docs)
- libstdc++-static (Fedora/Ubuntu)

**Recommended**: Set `llvm.download-ci-llvm = true` in bootstrap.toml to download prebuilt LLVM instead of building from source (saves hours).

## Testing

### Test Organization

- `tests/ui/` - UI tests (most common, test error messages and behavior)
- `tests/codegen/` - Tests for code generation quality
- `tests/debuginfo/` - Debug info generation tests
- `tests/mir-opt/` - MIR optimization tests
- `library/*/tests/` - Standard library unit tests
- `compiler/*/tests/` - Compiler unit tests

### Running Tests

```bash
# Run all tests (takes hours)
./x.py test

# Run specific test suite
./x.py test tests/ui

# Run single test
./x.py test tests/ui/test_name.rs

# Update test expectations (blessed)
./x.py test tests/ui --bless

# Test with specific stage
./x.py test --stage 1 tests/ui

# Test MLIR backend specifically
./x.py test compiler/rustc_codegen_mlir
```

## Toolchain

This fork uses **nightly-2025-12-05** as specified in `rust-toolchain.toml`. When working with this repository:

```bash
# The toolchain is automatically used when in this directory
rustup show

# Components are pre-configured:
# - rust-src (for -Z build-std)
# - rustc-dev (for compiler plugins)
# - llvm-tools-preview (for LLVM tools)
```

## Git Workflow

This is a fork with custom development. The current branch is `1.93.0-1`. Recent commits show:
- LLVM 22.0 integration
- Triton v3.6.0 integration
- Rust 1.93.0 base

When making changes:
1. Work in feature branches
2. Test thoroughly with `./x.py test`
3. Format code with `./x.py fmt`
4. Run clippy with `./x.py clippy`

### Commit Message Style

- Use conventional commit format: `type: subject`
- Common types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`
- Keep subject line concise and descriptive
- **Do not include Claude Code attribution** in commit messages

## MLIR Backend Specifics

The MLIR codegen backend (`compiler/rustc_codegen_mlir/`) is experimental and incomplete:

- Implements `CodegenBackend` trait
- Integrates with rustc's compilation pipeline
- Uses LLVM's MLIR infrastructure
- Still under active development (see TODOs in source)

When working on the MLIR backend:
- Refer to LLVM MLIR documentation
- Study existing `rustc_codegen_llvm` for patterns
- Use `rustc_codegen_ssa` for shared functionality
- Test against simple programs first

## Triton Integration

Triton compiler is embedded in `src/triton/`:
- Version 3.6.0
- Full Triton source tree
- Build with cmake/make (see `src/triton/README.md`)
- Used for GPU kernel compilation

## Building Triton from Source

Triton can be built standalone or automatically via the Rust build system. The Triton source is located in `src/triton/` (version 3.6.0).

### Dependencies

**Build Tools:**
- CMake >= 3.20
- Ninja >= 1.11.1
- Python 3 with setuptools, wheel
- pybind11 >= 2.13.1

**LLVM/MLIR:**
- LLVM version pinned in `src/triton/cmake/llvm-hash.txt`
- Can use parent Rust build's LLVM or build custom LLVM

**Optional:**
- NVIDIA CUDA toolkit (for NVIDIA backend)
- AMD ROCm (for AMD backend)

Install Python build dependencies:
```bash
pip install -r src/triton/python/requirements.txt
```

### Quick Build (Standalone)

Build Triton as a standalone Python module:

```bash
cd src/triton

# Install build dependencies
pip install -r python/requirements.txt

# Build and install in editable mode
pip install -e .
```

### Build via Rust Integration

Triton is automatically built when building the MLIR codegen backend:

```bash
# Triton is built as part of rustc_codegen_mlir
./x.py build compiler/rustc_codegen_mlir
```

The parent Rust build system configures Triton via `compiler/rustc_llvm/triton.toml`:
- Build type: Release
- Backends: amd, nvidia
- Python module: ON
- Proton profiler: OFF

### Custom LLVM Build

To build Triton with a custom LLVM installation:

```bash
cd src/triton

# Quick method (builds LLVM from source)
make dev-install-llvm

# Or manual method with custom LLVM paths
export LLVM_INCLUDE_DIRS=/path/to/llvm/include
export LLVM_LIBRARY_DIR=/path/to/llvm/lib
export LLVM_SYSPATH=/path/to/llvm
pip install -e .
```

### Build Optimization

Speed up Triton builds with these environment variables:

```bash
# Use clang/lld for faster linking
TRITON_BUILD_WITH_CLANG_LLD=true pip install -e .

# Enable ccache for incremental builds
TRITON_BUILD_WITH_CCACHE=true pip install -e .

# Limit parallel jobs (prevent out-of-memory)
MAX_JOBS=4 pip install -e .

# Skip build isolation (faster for development iterations)
pip install -e . --no-build-isolation

# Offline build mode (no network access)
TRITON_OFFLINE_BUILD=true pip install -e .
```

### Development Commands

Useful Make targets for Triton development:

```bash
cd src/triton

make dev-install        # Install with all dependencies
make test               # Run all tests
make test-nogpu         # Run tests without GPU
make test-lit           # Run MLIR lit tests
make test-cpp           # Run C++ unit tests
make test-python        # Run Python tests
make docs               # Build documentation
```

### Backend Configuration

Triton supports multiple GPU backends:

- **NVIDIA backend**: Generates NVPTX code, requires CUDA toolkit
- **AMD backend**: Generates AMDGPU code, requires ROCm
- Configurable via `TRITON_CODEGEN_BACKENDS` CMake variable

The Rust integration builds both backends by default.

### Key Environment Variables

Useful environment variables for Triton development:

| Variable | Purpose |
|----------|---------|
| `TRITON_HOME` | Custom cache directory (default: `~/.triton`) |
| `TRITON_BUILD_DIR` | Custom CMake build directory |
| `MLIR_ENABLE_DUMP` | Enable MLIR IR dumping for debugging |
| `TRITON_KERNEL_DUMP` | Dump kernel IR during compilation |
| `TRITON_INTERPRET` | Use Triton interpreter (no GPU required) |
| `TRITON_OFFLINE_BUILD` | Build without network access |
| `MAX_JOBS` | Limit parallel compilation jobs |
| `TRITON_BUILD_WITH_CCACHE` | Enable ccache for faster rebuilds |

### Build Outputs

After building:
- Python module: `src/triton/python/triton/`
- C++ libraries: `src/triton/lib/`
- Tools: `src/triton/bin/` (triton-opt, triton-reduce)
- Tests: `src/triton/test/`

When built via Rust integration, outputs go to `target/build/triton-build/build/`.

## Performance Considerations

- Initial build takes 1-2 hours (download prebuilt LLVM to speed up)
- Incremental builds are much faster with `--keep-stage`
- Use `./x.py check` instead of `build` during development
- Stage 1 is sufficient for most development; stage 2 for final testing
- LLVM assertions impact performance but help catch bugs

## Documentation Resources

- [Rust Compiler Dev Guide](https://rustc-dev-guide.rust-lang.org/) - Primary resource
- [Standard Library Dev Guide](https://std-dev-guide.rust-lang.org/)
- [CONTRIBUTING.md](CONTRIBUTING.md) - Contribution guidelines
- [INSTALL.md](INSTALL.md) - Detailed build instructions
- Ask questions at [Rust Zulip](https://rust-lang.zulipchat.com)

## Important Notes

- Never skip hooks (`--no-verify`, etc.) unless explicitly required
- Be careful with control flow changes (may affect profiling/debug info per LLVM's copilot-instructions)
- This repository is based on Rust 1.93.0, not the latest upstream
- MLIR backend and Triton integration are custom additions not in upstream Rust

<!-- bv-agent-instructions-v1 -->

---

## Beads Workflow Integration

This project uses [beads_viewer](https://github.com/Dicklesworthstone/beads_viewer) for issue tracking. Issues are stored in `.beads/` and tracked in git.

### Essential Commands

```bash
# View issues (launches TUI - avoid in automated sessions)
bv

# CLI commands for agents (use these instead)
bd ready              # Show issues ready to work (no blockers)
bd list --status=open # All open issues
bd show <id>          # Full issue details with dependencies
bd create --title="..." --type=task --priority=2
bd update <id> --status=in_progress
bd close <id> --reason="Completed"
bd close <id1> <id2>  # Close multiple issues at once
bd sync               # Commit and push changes
```

### Workflow Pattern

1. **Start**: Run `bd ready` to find actionable work
2. **Claim**: Use `bd update <id> --status=in_progress`
3. **Work**: Implement the task
4. **Complete**: Use `bd close <id>`
5. **Sync**: Always run `bd sync` at session end

### Key Concepts

- **Dependencies**: Issues can block other issues. `bd ready` shows only unblocked work.
- **Priority**: P0=critical, P1=high, P2=medium, P3=low, P4=backlog (use numbers, not words)
- **Types**: task, bug, feature, epic, question, docs
- **Blocking**: `bd dep add <issue> <depends-on>` to add dependencies

### Session Protocol

**Before ending any session, run this checklist:**

```bash
git status              # Check what changed
git add <files>         # Stage code changes
bd sync                 # Commit beads changes
git commit -m "..."     # Commit code
bd sync                 # Commit any new beads changes
git push                # Push to remote
```

### Best Practices

- Check `bd ready` at session start to find available work
- Update status as you work (in_progress → closed)
- Create new issues with `bd create` when you discover tasks
- Use descriptive titles and set appropriate priority/type
- Always `bd sync` before ending session

<!-- end-bv-agent-instructions -->

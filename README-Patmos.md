On nix and failing due to linker issues?

```zsh
export LD_LIBRARY_PATH=$(nix-build '<nixpkgs>' -A stdenv.cc.cc.lib)/lib:$LD_LIBRARY_PATH
```

How to build it?

Prerequisites:

```zsh
nix-shell -p cmake ninja rust python3
```

Prepare the Rustc:

```zsh
# Initialise the submodules.
git submodule update --init --recursive

# Todo: Add cc-rs Rust submodule? Attach guide?

# Check the Rust build system is working.
./x.py check

# Build Patmos target.
./x.py build --stage 1 library/core --target="patmos-unknown-none,aarch64-apple-darwin"
./x.py build --stage 1 library/std --target="patmos-unknown-none,aarch64-apple-darwin"
./x.py build --stage 1 src/tools/cargo --target="patmos-unknown-none,aarch64-apple-darwin"
# If you want to build the Miri interpreter as well
./x.py build --stage 1 src/tools/miri --target="patmos-unknown-none,aarch64-apple-darwin"
./x.py build --stage 1 --target="patmos-unknown-none,aarch64-apple-darwin"
```

Attach it to your toolchain:

```zsh
rustup toolchain link stage1-patmos build/host/stage1

rustc +stage1-patmos -Z unstable-options  --print target-spec-json --target patmos-unknown-none
```

How to add it to your cargo project:

```zsh
# In your project folder
rustup override set stage1-patmos
```

Edit your Cargo.toml:

```toml
[build]
target = "patmos-unknown-none"
```

## Troubleshooting

How to triage LLVM issues with Rust

```zsh
RUST_BACKTRACE=1 RUSTC_LOG=rustc_codegen_llvm=debug ./x.py build --stage 1 library/core --target patmos-unknown-none
```

How to rebuild the PML fast

```zsh
cmake --build build --target LLVMPatmosCodeGen
```

## Podman/Pre-CI setup

```zsh
podman machine stop 2>/dev/null
podman machine set --memory 8192
podman machine rm -f podman-machine-default 2>/dev/null
CONTAINERS_MACHINE_PROVIDER=applehv podman machine init -v /Volumes/SSD-99:/Volumes/SSD-99
CONTAINERS_MACHINE_PROVIDER=applehv podman machine start
```

```zsh
act --bind -j check --matrix os:ubuntu-latest -W .github/workflows/patmos-stage1.yml
```

Todo
Make a submodule of cc-rs and then update the Cargo.toml (ensuring to create update it) with the new path.
Document how to enable the CUSTOM_LINKER .env

## Rants

Holy fuck it is annoying that Rust needs to have 64-bit support, getting abstract ass errors as well.

Holy shit, pml, why the fuck has it been baked to C++/C's .main assembly object output buillshiiiit,

PML... PML... WHY PML...

entry-node-skip whyyyy, this shit aint'working, because it is making me insane, please PML handle this shit

file location to be in `/Volumes/SSD-99/Master_Thesis/template_patmos/pml_output.pml`
document how to make pml - always build the all of the libraries step by step

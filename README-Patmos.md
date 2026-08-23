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

### C++ link fails with "member of archive is not a bitcode file"

If `clang` fails with something like:

```
llvm-link: error: member of archive is not a bitcode file: 'clzsi2.c.obj'
```

Your `librt.a` is broken. The Patmos toolchain links C++/C code at the
bitcode level, so every object inside `librt.a` needs to be LLVM bitcode, not
a native ELF object.

This happens when compiler-rt gets built with the wrong toolchain pointed at
by `CMAKE_PROGRAM_PATH` — for example a `build-compiler-rt/CMakeCache.txt`
left over from configuring against a *different* LLVM checkout's
`build/bin`. When that happens, compiler-rt silently compiles to native
objects instead of bitcode, and copying the result into `librt.a` gives you
this exact error. (This took hours to resolve)

Check what's actually in `librt.a`:

```zsh
llvm-ar x build/patmos-unknown-unknown-elf/lib/librt.a clzsi2.c.obj
file clzsi2.c.obj
```

- `clzsi2.c.obj: LLVM IR bitcode` looks good, `librt.a` is fine, look elsewhere.
- `clzsi2.c.obj: ELF 32-bit MSB relocatable...` something 🅱️roke, re🅱️uild compiler-rt (and have a nice day)

In order to rebuild it correctly:

```zsh
rm -rf build-compiler-rt
mkdir build-compiler-rt && cd build-compiler-rt
cmake ../compiler-rt \
  -DCMAKE_TOOLCHAIN_FILE=../compiler-rt/cmake/patmos-clang-toolchain.cmake \
  -DCMAKE_PROGRAM_PATH="$(pwd)/../build/bin" \
  -DCOMPILER_RT_INCLUDE_TESTS=ON
make -j4
```

`CMAKE_PROGRAM_PATH` must point at *this same tree's* `build/bin` — not
another LLVM checkout's. That's the whole bug, every time.

Then install the result:

```zsh
cp lib/generic/libclang_rt.builtins-patmos.a \
   ../build/patmos-unknown-unknown-elf/lib/librt.a
```

Verify again with the `llvm-ar x` / `file` check above before moving on.
If you don't have time to debug why a given tree's compiler-rt build keeps
emitting native objects, you can copy a known-good, bitcode-verified
`librt.a` from another Patmos LLVM build of the same LLVM version, it's a
plain archive of target-independent bitcode, so it's portable across trees.
I wouldn't trust to rely on it long term, but it may help you just get the results if you
know it had worked before.

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

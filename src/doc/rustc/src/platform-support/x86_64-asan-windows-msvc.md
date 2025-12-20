# `x86_64-asan-windows-msvc`

**Tier: 3**

Target mirrors `x86_64-pc-windows-msvc` with AddressSanitizer enabled by default.

## Target maintainers

[@ChrisDenton](https://github.com/ChrisDenton)
[@dpaoliello](https://github.com/dpaoliello)
[@eholk](https://github.com/eholk)
[@Fulgen301](https://github.com/Fulgen301)
[@lambdageek](https://github.com/lambdageek)
[@sivadeilra](https://github.com/sivadeilra)
[@wesleywiser](https://github.com/wesleywiser)

## Requirements

This target is for cross-compilation only. Host tools are not supported because
this target's primary use cases do not require the host tools to be instrumented
with AddressSanitizer. The standard library is fully supported.

In all other aspects, this target is identical to `x86_64-pc-windows-msvc`.

## Building the target

This target can be built by adding it to the `target` list in `bootstrap.toml`.

```toml
[build]
target = ["x86_64-asan-windows-msvc"]
```

## Building Rust programs

Rust does not yet ship pre-compiled artifacts for this target. To compile for
this target, you will either need to build Rust with the target enabled (see
"Building the target" above), or build your own copy of `core` by using
`build-std` or similar.

Compilation can be done with:

```sh
rustc --target x86_64-asan-windows-msvc my_program.rs
```

## Testing

Programs compiled for this target require `clang_rt.asan_dynamic-x86_64.dll` to
be available. This can be installed by using the Visual Studio Installer to
install the C++ AddressSanitizer component. Once installed, add the directory
containing the DLL to your `PATH` or run your program from a Visual Studio
Developer Command Prompt.

## Cross-compilation toolchains and C code

Architectural cross-compilation from one Windows host to a different Windows
platform is natively supported by the MSVC toolchain provided the appropriate
components are selected when using the VS Installer.

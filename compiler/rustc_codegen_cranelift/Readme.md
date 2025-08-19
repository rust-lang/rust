# Cranelift codegen backend for rust

The goal of this project is to create an alternative codegen backend for the rust compiler based on [Cranelift](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift).
This has the potential to improve compilation times in debug mode.
If your project doesn't use any of the things listed under "Not yet supported", it should work fine.
If not please open an issue.

## Download using Rustup

The Cranelift codegen backend is distributed in nightly builds on Linux, macOS and x86_64 Windows. If you want to
install it using Rustup, you can do that by running:

```bash
$ rustup component add rustc-codegen-cranelift-preview --toolchain nightly
```

Once it is installed, you can enable it with one of the following approaches:
- `CARGO_PROFILE_DEV_CODEGEN_BACKEND=cranelift cargo +nightly build -Zcodegen-backend`
- Add the following to `.cargo/config.toml`:
    ```toml
    [unstable]
    codegen-backend = true

    [profile.dev]
    codegen-backend = "cranelift"
    ```
- Add the following to `Cargo.toml`:
    ```toml
    # This line needs to come before anything else in Cargo.toml
    cargo-features = ["codegen-backend"]

    [profile.dev]
    codegen-backend = "cranelift"
    ```

## Precompiled builds

You can also download a pre-built version from the [releases] page.
Extract the `dist` directory in the archive anywhere you want.
If you want to use `cargo clif build` instead of having to specify the full path to the `cargo-clif` executable, you can add the `bin` subdirectory of the extracted `dist` directory to your `PATH`.
(tutorial [for Windows](https://stackoverflow.com/a/44272417), and [for Linux/MacOS](https://unix.stackexchange.com/questions/26047/how-to-correctly-add-a-path-to-path/26059#26059)).

[releases]: https://github.com/rust-lang/rustc_codegen_cranelift/releases/tag/dev

## Building and testing

If you want to build the backend manually, you can download it from GitHub and build it yourself:

```bash
$ git clone https://github.com/rust-lang/rustc_codegen_cranelift
$ cd rustc_codegen_cranelift
$ ./y.sh build
```

To run the test suite replace the last command with:

```bash
$ ./y.sh prepare # only needs to be run the first time
$ ./test.sh
```

For more docs on how to build and test see [build_system/usage.txt](build_system/usage.txt) or the help message of `./y.sh`.

## Platform support

|OS \ architecture|x86\_64|AArch64|Riscv64|s390x (System-Z)|
|---|---|---|---|---|
|Linux|✅|✅|✅[^no-rustup]|✅[^no-rustup]|
|FreeBSD|✅[^no-rustup]|❓|❓|❓|
|AIX|❌[^xcoff]|N/A|N/A|❌[^xcoff]|
|Other unixes|❓|❓|❓|❓|
|macOS|✅|✅|N/A|N/A|
|Windows|✅|❌|N/A|N/A|

✅: Fully supported and tested
❓: Maybe supported, not tested
❌: Not supported at all

Not all targets are available as rustup component for nightly. See notes in the platform support matrix.

[^xcoff]: XCOFF object file format is not supported.
[^no-rustup]: Not available as [rustup component for nightly](https://rust-lang.github.io/rustup-components-history/). You can build it yourself.

## Usage

rustc_codegen_cranelift can be used as a near-drop-in replacement for `cargo build` or `cargo run` for existing projects.

Assuming `$cg_clif_dir` is the directory you cloned this repo into and you followed the instructions (`y.sh prepare` and `y.sh build` or `test.sh`).

In the directory with your project (where you can do the usual `cargo build`), run:

```bash
$ $cg_clif_dir/dist/cargo-clif build
```

This will build your project with rustc_codegen_cranelift instead of the usual LLVM backend.

For additional ways to use rustc_codegen_cranelift like the JIT mode see [usage.md](docs/usage.md).

## Building and testing with changes in rustc code

See [rustc_testing.md](docs/rustc_testing.md).

## Not yet supported

* SIMD ([tracked here](https://github.com/rust-lang/rustc_codegen_cranelift/issues/171), `std::simd` fully works, `std::arch` is partially supported)
* Unwinding on panics ([no cranelift support](https://github.com/bytecodealliance/wasmtime/issues/1677), `-Cpanic=abort` is enabled by default)

## License

Licensed under either of

  * Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
    http://www.apache.org/licenses/LICENSE-2.0)
  * MIT license ([LICENSE-MIT](LICENSE-MIT) or
    http://opensource.org/licenses/MIT)

at your option.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you shall be dual licensed as above, without any
additional terms or conditions.

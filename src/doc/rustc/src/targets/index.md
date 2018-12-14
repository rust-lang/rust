# Targets

`rustc` is a cross-compiler by default. This means that you can use any compiler to build for any
architecture. The list of *targets* are the possible architectures that you can build for.

To see all the options that you can set with a target, see the docs
[here](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_target/spec/struct.Target.html).

To compile to a particular target, use the `--target` flag:

```bash
$ rustc src/main.rs --target=wasm32-unknown-unknown
```

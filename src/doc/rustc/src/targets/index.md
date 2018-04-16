# Targets

`rustc` is a cross-compiler by default. This means that you can use any compiler to build for any
architecture. The list of *targets* are the possible architectures that you can build for.

You can see the API docs for a given target
[here](https://doc.rust-lang.org/nightly/nightly-rustc/rustc_back/target/struct.Target.html), all
of these options can be set on a per-target basis.

To compile to a particular target, use the `--target` flag:

```bash
$ rustc src/main.rs --target=wasm32-unknown-unknown
```
# Build distribution artifacts

You might want to build and package up the compiler for distribution.
You’ll want to run this command to do it:

```bash
./x dist
```

# Install from source

You might want to prefer installing Rust (and tools configured in your configuration)
by building from source. If so, you want to run this command:

```bash
./x install
```

   Note: If you are testing out a modification to a compiler, you might
   want to build the compiler (with `./x build`) then create a toolchain as
   discussed in [here][create-rustup-toolchain].

   For example, if the toolchain you created is called "foo", you would then
   invoke it with `rustc +foo ...` (where ... represents the rest of the arguments).

Instead of installing Rust (and tools in your config file) globally, you can set `DESTDIR`
environment variable to change the installation path. If you want to set installation paths
more dynamically, you should prefer [install options] in your config file to achieve that.

[create-rustup-toolchain]: ./how-to-build-and-run.md#creating-a-rustup-toolchain
[install options]: https://github.com/rust-lang/rust/blob/f7c8928f035370be33463bb7f1cd1aeca2c5f898/config.example.toml#L422-L442
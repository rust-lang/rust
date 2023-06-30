# Build distribution artifacts

You might want to build and package up the compiler for distribution.
You’ll want to run this command to do it:

```bash
./x dist
```

# Install distribution artifacts

If you’ve built a distribution artifact you might want to install it and
test that it works on your target system. You’ll want to run this command:

```bash
./x install
```

   Note: If you are testing out a modification to a compiler, you
   might want to use it to compile some project.
   Usually, you do not want to use `./x install` for testing.
   Rather, you should create a toolchain as discussed in
   [here][create-rustup-toolchain].

   For example, if the toolchain you created is called foo, you
   would then invoke it with `rustc +foo ...` (where ... represents
   the rest of the arguments).

[create-rustup-toolchain]: ./how-to-build-and-run.md#creating-a-rustup-toolchain
# Quickstart

This is a quickstart guide about getting the compiler running. For more
information on the individual steps, see the other pages in this chapter.

First, clone the repository:

```sh
git clone https://github.com/rust-lang/rust.git
cd rust
```

When building the compiler, we don't use `cargo` directly, instead we use a
wrapper called "x". It is invoked with `./x`.

We need to create a configuration for the build. Use `./x setup` to create a
good default.

```sh
./x setup
```

Then, we can build the compiler. Use `./x build` to build the compiler, standard
library and a few tools. You can also `./x check` to just check it. All these
commands can take specific components/paths as arguments, for example `./x check
compiler` to just check the compiler.

```sh
./x build
```

> When doing a change to the compiler that does not affect the way it compiles
the standard library (so for example, a change to an error message), use
`--keep-stage-std 1` to avoid recompiling it.

After building the compiler and standard library, you now have a working
compiler toolchain. You can use it with rustup by linking it.

```sh
rustup toolchain link stage1 build/host/stage1
```

Now you have a toolchain called `stage1` linked to your build. You can use it to
test the compiler.

```sh
rustc +stage1 testfile.rs
```

After doing a change, you can run the compiler test suite with `./x test`.

`./x test` runs the full test suite, which is slow and rarely what you want.
Usually, `./x test tests/ui` is what you want after a compiler change, testing
all [UI tests](../tests/ui.md) that invoke the compiler on a specific test file
and check the output.

```sh
./x test tests/ui
```

Use `--bless` if you've made a change and want to update the `.stderr` files
with the new output.

Congrats, you are now ready to make a change to the compiler! If you have more
questions, [the full chapter](./how-to-build-and-run.md) might contain the
answers, and if it doesn't, feel free to ask for help on
[Zulip](https://rust-lang.zulipchat.com/#narrow/stream/182449-t-compiler.2Fhelp).

If you use VSCode, Vim, Emacs or Helix, `./x setup` will ask you if you want to
set up the editor config. For more information, check out [suggested
workflows](./suggested.md).

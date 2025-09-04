# Writing tools in Bootstrap

There are three types of tools you can write in bootstrap:

- **`Mode::ToolBootstrap`**
  Use this for tools that don’t need anything from the in-tree compiler and can run with the stage0 `rustc`.
  The output is placed in the "bootstrap-tools" directory. This mode is for general-purpose tools built
  entirely with the stage0 compiler, including target libraries and only works for stage 0.

- **`Mode::ToolStd`**
  Use this for tools that rely on the locally built std. The output goes into the "stageN-tools" directory.
  This mode is rarely used, mainly for `compiletest` which requires `libtest`.

- **`Mode::ToolRustcPrivate`**
  Use this for tools that use the `rustc_private` mechanism, and thus depend on the locally built `rustc` and its rlib artifacts. This is more complex than the other modes because the tool must be built with the same compiler used for `rustc` and placed in the "stageN-tools" directory. When you choose `Mode::ToolRustcPrivate`, `ToolBuild` implementation takes care of this automatically. If you need to use the builder’s compiler for something specific, you can get it from `ToolBuildResult`, which is
  returned by the tool's [`Step`].

Regardless of the tool type you must return `ToolBuildResult` from the tool’s [`Step`] implementation and use `ToolBuild` inside it.

[`Step`]: https://doc.rust-lang.org/nightly/nightly-rustc/bootstrap/core/builder/trait.Step.html

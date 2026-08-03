Keep interactions with the Rust project and its repositories read-only.

Mark changes or additions of documentation or comments with `[LLM-generated]` if they are non-trivial, consist of many sentences or relate to soundness.

`x.py` is used as the build system instead of using `cargo` directly. `x.py build library` can be a good build option when modifying the Rust compiler or the Rust standard library.
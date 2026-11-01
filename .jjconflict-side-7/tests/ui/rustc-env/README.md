Some environment variables affect rustc's behavior not because they are major compiler interfaces
but rather because rustc is, ultimately, a Rust program, with debug logging, stack control, etc.

Prefer to group tests that use environment variables to control something about rustc's core UX,
like "can we parse this number of parens if we raise RUST_MIN_STACK?" with related code for that
compiler feature.

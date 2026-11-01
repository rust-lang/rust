`shim_utils` is a small library crate containing shared code that is used by
bootstrap's `rustc` and `rustdoc` shims, and also by bootstrap itself or other
bootstrap tools.

Using a dedicated crate should be faster than trying to link `build_helper`
or `bootstrap` into the shims.

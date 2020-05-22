A non-root module tried to import macros from another crate.

Example of erroneous code:

```compile_fail,E0468
mod foo {
    #[macro_use(debug_assert)]  // error: must be at crate root to import
    extern crate core;          //        macros from another crate
    fn run_macro() { debug_assert!(true); }
}
```

Only `extern crate` imports at the crate root level are allowed to import
macros.

Either move the macro import to crate root or do without the foreign macros.
This will work:

```
#[macro_use(debug_assert)] // ok!
extern crate core;

mod foo {
    fn run_macro() { debug_assert!(true); }
}
# fn main() {}
```

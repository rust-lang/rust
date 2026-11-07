// Check that `?Trait` matches the macro fragment specifier `ty`.
// Syntactically trait object types can be "bare" (i.e., lack the prefix `dyn`),
// even in newer editions like Rust 2021.
// Therefore the arm `?$Trait:path` shouldn't get reached.

//@ edition: 2021
//@ check-pass

macro_rules! check {
    ($Ty:ty) => {};
    (?$Trait:path) => { compile_error!("non-ty"); };
}

check! { ?Trait }

fn main() {}

// Check that unknown attribute error is shown even if there are unresolved macros.

#[marco_use] // typo
//~^ ERROR The attribute `marco_use` is currently unknown to the compiler
mod foo {
    macro_rules! bar {
        () => ();
    }
}

fn main() {
   bar!(); //~ ERROR cannot find macro `bar!` in this scope
}

// Check that unknown attribute error is shown even if there are unresolved macros.

#[marco_use] // typo
//~^ ERROR cannot find attribute `marco_use` in this scope
mod foo {
    macro_rules! bar {
        () => ();
    }
}

fn main() {
   bar!(); //~ ERROR cannot find macro `bar` in this scope
}

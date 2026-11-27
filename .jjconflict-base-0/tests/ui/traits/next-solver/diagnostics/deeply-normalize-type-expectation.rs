//@ compile-flags: -Znext-solver

// Make sure we try to mention a deeply normalized type in a type mismatch error.

trait Mirror {
    type Assoc;
}
impl<T> Mirror for T {
    type Assoc = T;
}

fn needs<T>(_: <T as Mirror>::Assoc) {}

fn main() {
    needs::<i32>(());
    //~^ ERROR mismatched types
}

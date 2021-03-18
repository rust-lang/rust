struct Reactor {
    input_cells: Vec<usize>,
}

impl Reactor {
    pub fn new() -> Self {
        input_cells: Vec::new()
        //~^ ERROR cannot find value `input_cells` in this scope
        //~| ERROR parenthesized type parameters may only be used with a `Fn` trait
        //~| ERROR missing generics for struct `Vec`
    }
}

// This case isn't currently being handled gracefully, including for completeness.
fn main() {}

struct Reactor {
    input_cells: Vec<usize>,
}

impl Reactor {
    pub fn new() -> Self {
        input_cells: Vec::new()
        //~^ ERROR cannot find value `input_cells` in this scope
        //~| ERROR parenthesized type parameters may only be used with a `Fn` trait
        //~| ERROR wrong number of type arguments: expected at least 1, found 0
    }
}

// This case isn't currently being handled gracefully, including for completeness.
fn main() {}

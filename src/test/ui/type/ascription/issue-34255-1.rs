struct Reactor {
    input_cells: Vec<usize>,
}

impl Reactor {
    pub fn new() -> Self {
        input_cells: Vec::new()
        //~^ ERROR cannot find value `input_cells` in this scope
        //~| ERROR parenthesized type parameters may only be used with a `Fn` trait
        //~| ERROR wrong number of type arguments: expected 1, found 0
        //~| WARNING this was previously accepted by the compiler but is being phased out
    }
}

// This case isn't currently being handled gracefully, including for completeness.

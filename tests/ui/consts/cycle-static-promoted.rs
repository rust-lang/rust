struct Value {
    values: &'static [&'static Value],
}

// This `static` recursively points to itself through a promoted (the slice).
static VALUE: Value = Value {
    values: &[&VALUE],
    //~^ ERROR: cycle detected when evaluating initializer of static `VALUE`
};

fn main() {}

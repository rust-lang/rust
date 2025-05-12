//@ check-pass

struct Value {
    values: &'static [&'static Value],
}

// This `static` recursively points to itself through a promoted (the slice).
static VALUE: Value = Value {
    values: &[&VALUE],
};

fn main() {}

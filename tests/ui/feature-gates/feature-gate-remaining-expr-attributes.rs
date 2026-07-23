fn free_function() -> usize {
    0
}

struct Value {
    field: usize,
}

impl Value {
    fn method(self) -> Self {
        self
    }
}

fn main() {
    let value = Value { field: 0 };

    let _ = #[allow(unused)] 42;
    //~^ ERROR attributes on expressions are experimental

    let _ = #[allow(unused)] free_function();
    //~^ ERROR attributes on expressions are experimental

    let _ = #[allow(unused)] value.method().field;
    //~^ ERROR attributes on expressions are experimental

    let _ = #[allow(unused)] [0][0];
    //~^ ERROR attributes on expressions are experimental

    let _ = #[allow(unused)] 1 + 2;
    //~^ ERROR attributes on expressions are experimental
}

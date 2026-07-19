//@ check-pass

struct Value;

impl Value {
    fn first(self) -> Self {
        self
    }

    fn second(self) -> usize {
        0
    }
}

fn main() {
    let _ = #[allow(unused)] Value.first();
    let _ = #[allow(unused)] Value.first().second();
    let _ = #[allow(unused)] (Value.first().second());
}

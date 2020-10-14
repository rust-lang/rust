// Type arguments in unresolved entities (reporting errors before type checking)
// should have their types recorded.

trait Tr<T> {}

fn closure() {
    let _ = |a, b: _| -> _ { 0 }; //~ ERROR type annotations needed
}

fn main() {}

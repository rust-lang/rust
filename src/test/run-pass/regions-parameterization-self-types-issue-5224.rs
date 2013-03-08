// Test how region-parameterization inference
// interacts with explicit self types.
//
// Issue #5224.

trait Getter {
    // This trait does not need to be
    // region-parameterized, because 'self
    // is bound in the self type:
    fn get(&self) -> &'self int;
}

struct Foo {
    field: int
}

impl Getter for Foo {
    fn get(&self) -> &'self int { &self.field }
}

fn get_int<G: Getter>(g: &G) -> int {
    *g.get()
}

fn main() {
    let foo = Foo { field: 22 };
    fail_unless!(get_int(&foo) == 22);
}

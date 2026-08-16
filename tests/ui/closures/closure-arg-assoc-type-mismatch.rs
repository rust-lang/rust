// When a closure is passed as a function argument and the E0271 projection
// mismatch is about an associated type of the closure's return type, the error
// should point at the return expression inside the closure and label the
// closure declaration — mirroring the FnOnce-output diagnostic.
// https://github.com/rust-lang/rust/issues/42390

trait MyTrait {
    type Item;
}

struct S;
impl MyTrait for S {
    type Item = u32;
}

// Bound on the closure's output type's associated type.
fn needs_unit_item<F: Fn() -> S>(_f: F)
where
    F::Output: MyTrait<Item = ()>,
{
}

fn main() {
    needs_unit_item(|| S); //~ ERROR type mismatch resolving `<S as MyTrait>::Item == ()`

    needs_unit_item(|| { S }); //~ ERROR type mismatch resolving `<S as MyTrait>::Item == ()`

    let x = S;
    needs_unit_item(|| { let _ = 0; x }); //~ ERROR type mismatch resolving `<S as MyTrait>::Item == ()`
}

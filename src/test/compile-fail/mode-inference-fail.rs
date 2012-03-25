// In this test, the mode gets inferred to ++ due to the apply_int(),
// but then we get a failure in the generic apply().

fn apply<A>(f: fn(A) -> A, a: A) -> A { f(a) }
fn apply_int(f: fn(int) -> int, a: int) -> int { f(a) }

fn main() {
    let f = {|i| i};
    assert apply_int(f, 2) == 2;
    assert apply(f, 2) == 2; //! ERROR expected argument mode ++
}

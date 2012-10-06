trait TraitA {
    fn method_a() -> int;
}

trait TraitB {
    fn gimme_an_a<A: TraitA>(a: A) -> int;
}

impl int: TraitB {
    fn gimme_an_a<A: TraitA>(a: A) -> int {
        a.method_a() + self
    }
}

fn call_it<B: TraitB>(b: B)  -> int {
    let y = 4u;
    b.gimme_an_a(y) //~ ERROR failed to find an implementation of trait @TraitA
}

fn main() {
    let x = 3i;
    assert call_it(x) == 22;
}

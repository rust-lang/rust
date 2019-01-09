trait TraitA {
    fn method_a(&self) -> isize;
}

trait TraitB {
    fn gimme_an_a<A:TraitA>(&self, a: A) -> isize;
}

impl TraitB for isize {
    fn gimme_an_a<A:TraitA>(&self, a: A) -> isize {
        a.method_a() + *self
    }
}

fn call_it<B:TraitB>(b: B)  -> isize {
    let y = 4;
    b.gimme_an_a(y) //~ ERROR `{integer}: TraitA` is not satisfied
}

fn main() {
    let x = 3;
    assert_eq!(call_it(x), 22);
}

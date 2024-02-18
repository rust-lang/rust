//@ check-pass

trait TraitA {
    const VALUE: usize;
}

struct A;
impl TraitA for A {
    const VALUE: usize = 1;
}

trait TraitB {
    type MyA: TraitA;
    const VALUE: usize = Self::MyA::VALUE;
}

struct B;
impl TraitB for B {
    type MyA = A;
}

fn main() {
    let _ = [0; A::VALUE];
    let _ = [0; B::VALUE]; // Indirectly refers to `A::VALUE`
}

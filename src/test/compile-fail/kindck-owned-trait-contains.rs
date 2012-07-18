iface repeat<A> { fn get() -> A; }

impl<A:copy> of repeat<A> for @A {
    fn get() -> A { *self }
}

fn repeater<A:copy>(v: @A) -> repeat<A> {
    // Note: owned kind is not necessary as A appears in the iface type
    v as repeat::<A> // No
}

fn main() {
    // Here, an error results as the type of y is inferred to
    // repeater<&lt/3> where lt is the block.
    let y = { //~ ERROR reference is not valid outside of its lifetime
        let x = &3;
        repeater(@x)
    };
    assert 3 == *(y.get()); //~ ERROR reference is not valid outside of its lifetime
}
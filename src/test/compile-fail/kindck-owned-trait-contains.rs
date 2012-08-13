trait repeat<A> { fn get() -> A; }

impl<A:copy> @A: repeat<A> {
    fn get() -> A { *self }
}

fn repeater<A:copy>(v: @A) -> repeat<A> {
    // Note: owned kind is not necessary as A appears in the trait type
    v as repeat::<A> // No
}

fn main() {
    // Here, an error results as the type of y is inferred to
    // repeater<&lt/3> where lt is the block.
    let y = {
        let x: &blk/int = &3; //~ ERROR cannot infer an appropriate lifetime
        repeater(@x)
    };
    assert 3 == *(y.get());
}
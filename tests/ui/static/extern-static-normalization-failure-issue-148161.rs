// Regression test for https://github.com/rust-lang/rust/issues/148161
trait Trait {
    type Assoc;
}

impl Trait for u8 {
    type Assoc = i8;
}

struct Struct<T: Trait> {
    member: T::Assoc,
}

unsafe extern "C" {
    // This used to be an ICE due to normalization failure on `<Struct<i8> as Trait>::Assoc`
    // while wf checking this static item.
    static VAR: Struct<i8>;
    //~^ ERROR: the trait bound `i8: Trait` is not satisfied
}

fn main() {}

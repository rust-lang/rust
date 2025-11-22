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
    static VAR: Struct<i8>;
    //~^ ERROR: the trait bound `i8: Trait` is not satisfied
}

fn main() {}

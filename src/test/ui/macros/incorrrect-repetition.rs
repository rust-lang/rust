#![deny(incorrect_macro_fragment_repetition)]

macro_rules! sneaky {
    ($($i:ident $e:expr)*) => {}
    //~^ ERROR `$e:expr` is followed (through repetition) by `$i:ident`, which is not allowed for
    //~| WARN this was previously accepted by the compiler but is being phased out
}

fn main() {
    sneaky!(a b c d);
}

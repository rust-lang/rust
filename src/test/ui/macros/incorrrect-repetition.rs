macro_rules! sneaky {
    ($($i:ident $e:expr)*) => {}
    //~^ ERROR `$e:expr` is followed (through repetition) by `$i:ident`, which is not allowed for
}

fn main() {
    sneaky!(a b c d);
}

macro_rules! sneaky {
    ($($i:ident $e:expr)*) => {}
    //~^ WARN `$e:expr` is followed (through repetition) by `$i:ident`, which is not allowed for
}

fn main() {
    sneaky!(a b c d);
    let x: () = 1;
    //~^ ERROR
}

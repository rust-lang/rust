// https://github.com/rust-lang/rust/issues/29383
enum E {
    A,
    B,
}

fn main() {
    match None {
        None => {}
        Some(E::A(..)) => {}
        //~^ ERROR expected tuple struct or tuple variant, found unit variant `E::A`
        Some(E::B(..)) => {}
        //~^ ERROR expected tuple struct or tuple variant, found unit variant `E::B`
    }
}

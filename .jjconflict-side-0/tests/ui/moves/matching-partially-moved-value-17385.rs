// https://github.com/rust-lang/rust/issues/17385
struct X(isize);

enum Enum {
    Variant1,
    Variant2
}

impl Drop for X {
    fn drop(&mut self) {}
}
impl Drop for Enum {
    fn drop(&mut self) {}
}

fn main() {
    let foo = X(1);
    drop(foo);
    match foo { //~ ERROR use of moved value
        X(1) => (),
        _ => unreachable!()
    }

    let e = Enum::Variant2;
    drop(e);
    match e { //~ ERROR use of moved value
        Enum::Variant1 => unreachable!(),
        Enum::Variant2 => ()
    }
}

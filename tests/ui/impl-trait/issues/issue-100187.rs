//@ check-pass

trait Trait<T> {
    type Ty;
}
impl Trait<&u8> for () {
    type Ty = ();
}

fn test<'a, 'b>() -> impl Trait<&'a u8, Ty = impl Sized + 'b> {}

fn main() {}

// check-pass

pub struct Struct {}

pub trait Trait<'a> {
    type Assoc;

    fn method() -> Self::Assoc;
}

impl<'a> Trait<'a> for Struct {
    type Assoc = ();

    fn method() -> Self::Assoc {}
}

pub fn function<F, T>(f: F)
where
    F: for<'a> FnOnce(<T as Trait<'a>>::Assoc),
    T: for<'b> Trait<'b>,
{
    f(T::method());
}

fn main() {
    function::<_, Struct>(|_| {});
}

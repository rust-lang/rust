pub struct Def;

pub trait Abc { }

impl<T> Abc for Option<T> { }

impl Abc for Def { }

impl<T> Abc for Vec<T> { }

impl Clone for Def {
    fn clone(&self) -> Def {
        Def
    }
}

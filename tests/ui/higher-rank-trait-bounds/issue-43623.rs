// check-pass

pub trait Trait<'a> {
    type Assoc;
}

pub struct Type;

impl<'a> Trait<'a> for Type {
    type Assoc = ();
}

pub fn break_me<T, F>(f: F)
where
    T: for<'b> Trait<'b>,
    F: for<'b> FnMut(<T as Trait<'b>>::Assoc),
{
    break_me::<Type, fn(_)>;
}

fn main() {}

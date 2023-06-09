// check-pass

use std::borrow::Cow;

enum _Recursive<'a>
where
    Self: ToOwned<Owned=Box<Self>>
{
    Variant(MyCow<'a, _Recursive<'a>>),
}

pub struct Wrapper<T>(T);

pub struct MyCow<'a, T: ToOwned<Owned=Box<T>> + 'a>(Wrapper<Cow<'a, T>>);

fn main() {}

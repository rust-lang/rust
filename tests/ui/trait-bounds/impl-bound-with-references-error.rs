// Regression test for #105138.
// This test ensures that the compiler does not add note
// for implementation of trait whose inner type is erroneous.

pub enum LabelText {
    Plain,
}

impl<T> From<T> for LabelText
//~^ ERROR conflicting implementations of trait `From<LabelText>` for type `LabelText` [E0119]
where
    T: Into<Cow<'static, str>>,
    //~^ ERROR cannot find type `Cow` in this scope [E0412]
{
    fn from(text: T) -> Self {
        LabelText::Plain(text.into()) //~ ERROR expected function, found `LabelText`
    }
}

fn main() {}

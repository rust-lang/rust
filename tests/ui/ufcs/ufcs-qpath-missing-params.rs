use std::borrow::Cow;

pub trait IntoCow<'a, B: ?Sized> where B: ToOwned {
    fn into_cow(self) -> Cow<'a, B>;
}

impl<'a> IntoCow<'a, str> for String {
    fn into_cow(self) -> Cow<'a, str> {
        Cow::Owned(self)
    }
}

fn main() {
    <String as IntoCow>::into_cow("foo".to_string());
      //~^ ERROR missing generics for

    <String as IntoCow>::into_cow::<str>("foo".to_string());
    //~^ ERROR method takes 0 generic arguments but 1
    //~| ERROR missing generics for
}

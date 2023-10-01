#![warn(clippy::from_over_into)]

struct InMacro(String);

macro_rules! in_macro {
    () => {
        Self::new()
    };
}

impl Into<InMacro> for String {
    //~^ ERROR: an implementation of `From` is preferred since it gives you `Into<_>` for free
    fn into(self) -> InMacro {
        InMacro(in_macro!())
    }
}

struct WeirdUpperSelf;

impl Into<WeirdUpperSelf> for &'static [u8] {
    //~^ ERROR: an implementation of `From` is preferred since it gives you `Into<_>` for free
    fn into(self) -> WeirdUpperSelf {
        let _ = Self::default();
        WeirdUpperSelf
    }
}

struct ContainsVal;

impl Into<u8> for ContainsVal {
    //~^ ERROR: an implementation of `From` is preferred since it gives you `Into<_>` for free
    fn into(self) -> u8 {
        let val = 1;
        val + 1
    }
}

pub struct Lval<T>(T);

pub struct Rval<T>(T);

impl<T> Into<Rval<Self>> for Lval<T> {
    //~^ ERROR: an implementation of `From` is preferred since it gives you `Into<_>` for free
    fn into(self) -> Rval<Self> {
        Rval(self)
    }
}

fn main() {}

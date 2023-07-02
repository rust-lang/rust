#![warn(clippy::from_over_into)]

struct InMacro(String);

macro_rules! in_macro {
    () => {
        Self::new()
    };
}

impl Into<InMacro> for String {
    fn into(self) -> InMacro {
        InMacro(in_macro!())
    }
}

struct WeirdUpperSelf;

impl Into<WeirdUpperSelf> for &'static [u8] {
    fn into(self) -> WeirdUpperSelf {
        let _ = Self::default();
        WeirdUpperSelf
    }
}

struct ContainsVal;

impl Into<u8> for ContainsVal {
    fn into(self) -> u8 {
        let val = 1;
        val + 1
    }
}

pub struct Lval<T>(T);

pub struct Rval<T>(T);

impl<T> Into<Rval<Self>> for Lval<T> {
    fn into(self) -> Rval<Self> {
        Rval(self)
    }
}

fn main() {}

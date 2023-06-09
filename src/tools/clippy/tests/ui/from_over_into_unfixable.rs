#![warn(clippy::from_over_into)]

struct InMacro(String);

macro_rules! in_macro {
    ($e:ident) => {
        $e
    };
}

impl Into<InMacro> for String {
    fn into(self) -> InMacro {
        InMacro(in_macro!(self))
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

fn main() {}

// ignore-tidy-linelength


impl u8 {
//~^ error: only a single inherent implementation marked with `#[lang = "u8"]` is allowed for the `u8` primitive
    pub const B: u8 = 0;
}

impl str {
//~^ error: only a single inherent implementation marked with `#[lang = "str"]` is allowed for the `str` primitive
    fn foo() {}
    fn bar(self) {}
}

impl char {
//~^ error: only a single inherent implementation marked with `#[lang = "char"]` is allowed for the `char` primitive
    pub const B: u8 = 0;
    pub const C: u8 = 0;
    fn foo() {}
    fn bar(self) {}
}

fn main() {}

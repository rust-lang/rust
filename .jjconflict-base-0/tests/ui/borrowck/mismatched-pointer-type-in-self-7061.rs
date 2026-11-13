// https://github.com/rust-lang/rust/issues/7061
//@ dont-require-annotations: NOTE

struct BarStruct;

impl<'a> BarStruct {
    fn foo(&'a mut self) -> Box<BarStruct> { self }
    //~^ ERROR mismatched types
    //~| NOTE expected struct `Box<BarStruct>`
    //~| NOTE found mutable reference `&'a mut BarStruct`
}

fn main() {}

struct BarStruct;

impl<'a> BarStruct {
    fn foo(&'a mut self) -> Box<BarStruct> { self }
    //~^ ERROR mismatched types
    //~| expected struct `std::boxed::Box<BarStruct>`
    //~| found mutable reference `&'a mut BarStruct`
}

fn main() {}

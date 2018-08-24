struct BarStruct;

impl<'a> BarStruct {
    fn foo(&'a mut self) -> Box<BarStruct> { self }
    //~^ ERROR mismatched types
    //~| expected type `std::boxed::Box<BarStruct>`
    //~| found type `&'a mut BarStruct`
}

fn main() {}

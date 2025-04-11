struct BarStruct;

impl<'a> BarStruct {
    fn foo(&'a mut self) -> Box<BarStruct> { self }
    //~^ ERROR mismatched types
    //~| NOTE_NONVIRAL expected struct `Box<BarStruct>`
    //~| NOTE_NONVIRAL found mutable reference `&'a mut BarStruct`
}

fn main() {}

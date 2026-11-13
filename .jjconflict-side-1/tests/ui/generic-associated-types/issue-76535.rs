pub trait SubTrait {}

pub trait SuperTrait {
    type SubType<'a>: SubTrait where Self: 'a;

    fn get_sub<'a>(&'a mut self) -> Self::SubType<'a>;
}

pub struct SubStruct<'a> {
    sup: &'a mut SuperStruct,
}

impl<'a> SubTrait for SubStruct<'a> {}

pub struct SuperStruct {
    value: u8,
}

impl SuperStruct {
    pub fn new(value: u8) -> SuperStruct {
        SuperStruct { value }
    }
}

impl SuperTrait for SuperStruct {
    type SubType<'a> = SubStruct<'a>;

    fn get_sub<'a>(&'a mut self) -> Self::SubType<'a> {
        SubStruct { sup: self }
    }
}

fn main() {
    let sub: Box<dyn SuperTrait<SubType = SubStruct>> = Box::new(SuperStruct::new(0));
      //~^ ERROR missing generics for associated type
      //~| ERROR the trait
}

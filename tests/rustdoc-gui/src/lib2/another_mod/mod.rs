pub struct MultiImplBlockStruct;

impl MultiImplBlockStruct {
    pub fn first_fn() {}
}

impl MultiImplBlockStruct {
    pub fn second_fn(self) -> bool { true }
}

pub trait MultiImplBlockTrait {
    fn first_fn();
    fn second_fn(self) -> u32;
}

impl MultiImplBlockTrait for MultiImplBlockStruct {
    fn first_fn() {}
    fn second_fn(self) -> u32 { 1 }
}

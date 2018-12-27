#[derive(Clone)]
pub struct MaskedStruct;

pub trait MaskedTrait {
    fn masked_method();
}

impl MaskedTrait for String {
    fn masked_method() {}
}

pub struct CustomEq;

impl Eq for CustomEq {}
impl PartialEq for CustomEq {
    fn eq(&self, _: &Self) -> bool {
        false
    }
}

pub const NONE: Option<CustomEq> = None;
pub const SOME: Option<CustomEq> = Some(CustomEq);

pub trait AssocConst {
    const NONE: Option<CustomEq> = None;
    const SOME: Option<CustomEq> = Some(CustomEq);
}

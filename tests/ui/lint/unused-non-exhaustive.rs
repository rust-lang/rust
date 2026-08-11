#![deny(unused_attributes)]

#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` has no effect on an unreachable item
struct PrivateUnitStruct;

#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` has no effect on an unreachable item
struct PrivateStructWithPrivateField {
    field: (),
}

#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` has no effect on an unreachable item
enum PrivateEnum {
    Variant { field: () },
}

enum PrivateVariant {
    #[non_exhaustive]
    //~^ ERROR `#[non_exhaustive]` has no effect on an unreachable item
    Variant { field: () },
}

#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` has no effect on a struct with non-public fields
pub struct PublicStructWithPrivateField {
    pub public: (),
    private: (),
}

#[non_exhaustive]
//~^ ERROR `#[non_exhaustive]` has no effect on a struct with non-public fields
pub struct PublicTupleStructWithPrivateField(pub (), ());

#[non_exhaustive]
pub struct PublicStructWithPublicField {
    pub field: (),
}

#[non_exhaustive]
pub enum PublicEnum {
    #[non_exhaustive]
    Variant { field: () },
}

fn main() {}

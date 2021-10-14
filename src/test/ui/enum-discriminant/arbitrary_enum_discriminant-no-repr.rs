#![crate_type="lib"]
#![feature(arbitrary_enum_discriminant)]

enum Enum {
//~^ ERROR `#[repr(inttype)]` must be specified
  Unit = 1,
  Tuple() = 2,
  Struct{} = 3,
}

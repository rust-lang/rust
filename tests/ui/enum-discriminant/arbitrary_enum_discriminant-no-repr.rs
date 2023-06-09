#![crate_type="lib"]

enum Enum {
//~^ ERROR `#[repr(inttype)]` must be specified
  Unit = 1,
  Tuple() = 2,
  Struct{} = 3,
}

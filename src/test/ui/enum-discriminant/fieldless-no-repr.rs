// check-pass
#![crate_type="lib"]

enum Enum {
  Unit = 1,
  Tuple() = 2,
  Struct{} = 3,
}

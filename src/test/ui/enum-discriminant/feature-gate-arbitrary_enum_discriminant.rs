#![crate_type="lib"]

enum Enum {
  Unit = 1,
  //~^ ERROR custom discriminant values are not allowed in enums with tuple or struct variants
  Tuple() = 2,
  //~^ ERROR discriminants on non-unit variants are experimental
  Struct{} = 3,
  //~^ ERROR discriminants on non-unit variants are experimental
}

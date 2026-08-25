fn main() {
  foo::/* definitely not harmful comment */(123, "foo") -> (u32); //~ ERROR: expected type, found `123`
  //~^ NOTE: while parsing this parenthesized list of type arguments starting
  //~^^ NOTE: expected type
}

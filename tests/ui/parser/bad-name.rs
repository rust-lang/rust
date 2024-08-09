fn main() {
  let x.y::<isize>.z foo;
  //~^ error: field expressions cannot have generic arguments
  //~| error: expected a pattern, found an expression
  //~| error: expected one of `(`, `.`, `::`, `:`, `;`, `=`, `?`, `|`, or an operator, found `foo`
}

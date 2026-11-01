fn main() {
  foo::( //~ HELP: consider removing the `::` here to call the expression
    //~^ NOTE: while parsing this parenthesized list of type arguments starting
    bar(x, y, z),
    bar(x, y, z),
    bar(x, y, z),
    bar(x, y, z),
    bar(x, y, z),
    bar(x, y, z),
    bar(x, y, z),
    baz("test"), //~ ERROR: expected type, found `"test"`
    //~^ NOTE: expected type
  )
}

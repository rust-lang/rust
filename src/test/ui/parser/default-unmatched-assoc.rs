fn main() {}

trait Foo {
    default!(); //~ ERROR cannot find macro `default` in this scope
    default do
    //~^ ERROR unmatched `default`
    //~| ERROR non-item in item list
}

struct S;
impl S {
    default!(); //~ ERROR cannot find macro `default` in this scope
    default do
    //~^ ERROR unmatched `default`
    //~| ERROR non-item in item list
}

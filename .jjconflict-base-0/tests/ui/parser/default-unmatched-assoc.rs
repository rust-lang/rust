fn main() {}

trait Foo {
    default!(); //~ ERROR cannot find macro `default` in this scope
    default do
    //~^ ERROR `default` is not followed by an item
    //~| ERROR non-item in item list
}

struct S;
impl S {
    default!(); //~ ERROR cannot find macro `default` in this scope
    default do
    //~^ ERROR `default` is not followed by an item
    //~| ERROR non-item in item list
}

// check-fail

trait Foo {
    type Assoc<'a, 'b>;
}
impl Foo for () {
    type Assoc<'a, 'b> = () where 'a: 'b;
    //~^ impl has stricter requirements than trait
}

fn main() {}

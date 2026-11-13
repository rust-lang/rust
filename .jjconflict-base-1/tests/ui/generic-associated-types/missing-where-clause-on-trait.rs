//@ check-fail

trait Foo {
    type Assoc<'a, 'b>;
}
impl Foo for () {
    type Assoc<'a, 'b> = () where 'a: 'b;
    //~^ ERROR impl has stricter requirements than trait
}

fn main() {}

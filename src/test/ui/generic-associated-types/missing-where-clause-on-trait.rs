// check-fail

#![feature(generic_associated_types)]

trait Foo {
    type Assoc<'a, 'b>;
}
impl Foo for () {
    type Assoc<'a, 'b> where 'a: 'b = ();
    //~^ `impl` associated type
}

fn main() {}

#![feature(box_syntax)]

fn id<T>(x: T) -> T { x }

trait Foo { }

impl<'a> Foo for &'a isize { }

fn main() {
    let blah;
    {
        let ss: &isize = &id(1);
        //~^ ERROR borrowed value does not live long enough
        blah = box ss as Box<Foo>;
    }
}

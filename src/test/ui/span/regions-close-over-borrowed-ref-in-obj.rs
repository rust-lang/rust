#![feature(box_syntax)]

fn id<T>(x: T) -> T { x }

trait Foo { }

impl<'a> Foo for &'a isize { }

fn main() {
    let blah;
    {
        let ss: &isize = &id(1);
        //~^ ERROR temporary value dropped while borrowed
        blah = box ss as Box<dyn Foo>;
    }
}

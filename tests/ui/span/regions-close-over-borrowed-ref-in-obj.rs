fn id<T>(x: T) -> T { x }

trait Foo { }

impl<'a> Foo for &'a isize { }

fn main() {

    let blah;

    {
        let ss: &isize = &id(1);
        //~^ ERROR temporary value dropped while borrowed
        blah = Box::new(ss) as Box<dyn Foo>;
    }
}

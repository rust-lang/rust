#![feature(box_syntax)]
#![feature(nll)]

trait Foo { fn get(&self); }

impl<A> Foo for A {
    fn get(&self) { }
}

fn main() {
    let _ = {
        let tmp0 = 3;
        let tmp1 = &tmp0;
        box tmp1 as Box<Foo + '_>
    };
    //~^^^ ERROR `tmp0` does not live long enough
}

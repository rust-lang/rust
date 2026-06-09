// Test for what happens when a type parameter `A` is closed over into
// an object. This should yield errors unless `A` (and the object)
// both have suitable bounds.

trait Foo { fn get(&self); }

impl<A> Foo for A {
    fn get(&self) {
    }
}

fn repeater3<'a,A:'a>(v: A) -> Box<dyn Foo + 'a> {
    Box::new(v) as Box<dyn Foo+'a>
}

fn main() {

    // Error results because the type of is inferred to be
    // ~Repeat<&'blk isize> where blk is the lifetime of the block below.

    let _ = {
        let tmp0 = 3;
        let tmp1 = &tmp0;
        repeater3(tmp1)
    };
    //~^^^ ERROR `tmp0` does not live long enough
}

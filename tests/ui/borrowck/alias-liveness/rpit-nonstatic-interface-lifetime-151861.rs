// Similar to other tests with lifetimes in the trait bounds of an `impl Trait`,
// but this shows that the lifetime in the outlive bound does not need to
// be `'static`, it can just be some unrelated lifetime. However, the lifetime
// in the trait bound must still be considered live.

#![forbid(unsafe_code)]

static STATIC: &str = "static";

trait Produce<'a> {
    fn produce(&self) -> &'a str;
}

struct Wrapper;

impl<'a> Produce<'a> for Wrapper {
    fn produce(&self) -> &'a str {
        STATIC
    }
}

fn make<'r, 'b>(_: &'r str, _: &'b ()) -> impl Produce<'r> + 'b {
    Wrapper
}

fn leak_it<T: 'static>(val: T) -> &'static T {
    Box::leak(Box::new(val))
}

fn main() {
    let b = ();
    let out;
    {
        let s = String::from("hi");
        let p = make(&s, &b);
        //~^ ERROR `s` does not live long enough
        out = p.produce();
    }
    // requires `&'r str: 'static`
    let _leaked = leak_it(out);
}

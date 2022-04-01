// check that static methods don't get to assume their trait-ref
// is well-formed.
// FIXME(#27579): this is just a bug. However, our checking with
// static inherent methods isn't quite working - need to
// fix that before removing the check.

trait Foo<'a, 'b, T>: Sized {
    fn make_me() -> Self { loop {} }
    fn static_evil(u: &'b u32) -> &'a u32;
}

struct Evil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b, Evil<'a, 'b>> for () {
    fn make_me() -> Self { }
    fn static_evil(u: &'b u32) -> &'a u32 {
        u
        //~^ ERROR lifetime may not live long enough
    }
}

struct IndirectEvil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b, ()> for IndirectEvil<'a, 'b> {
    fn make_me() -> Self { IndirectEvil(None) }
    fn static_evil(u: &'b u32) -> &'a u32 {
        let me = Self::make_me();
        //~^ ERROR lifetime may not live long enough
        loop {} // (`me` could be used for the lifetime transmute).
    }
}

impl<'a, 'b> Evil<'a, 'b> {
    fn inherent_evil(u: &'b u32) -> &'a u32 {
        u
        //~^ ERROR lifetime may not live long enough
    }
}

// while static methods don't get to *assume* this, we still
// *check* that they hold.

fn evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <()>::static_evil(b)
    //~^ ERROR lifetime may not live long enough
}

fn indirect_evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <IndirectEvil>::static_evil(b)
    //~^ ERROR lifetime may not live long enough
}

fn inherent_evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <Evil>::inherent_evil(b)
    //~^ ERROR lifetime may not live long enough
}


fn main() {}

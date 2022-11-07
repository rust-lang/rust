// check that associated consts can assume the impl header is well-formed.

// FIXME(aliemjay): we should check the impl header is WF at the use site
// but we currently don't in some cases. This is *unsound*.

trait Foo<'a, 'b, T>: Sized {
    const EVIL: fn(u: &'b u32) -> &'a u32;
}

struct Evil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b, Evil<'a, 'b>> for () {
    const EVIL: fn(&'b u32) -> &'a u32 = { |u| u };
}

struct IndirectEvil<'a, 'b: 'a>(Option<&'a &'b ()>);

impl<'a, 'b> Foo<'a, 'b, ()> for IndirectEvil<'a, 'b> {
    const EVIL: fn(&'b u32) -> &'a u32 = { |u| u };
}

impl<'a, 'b> Evil<'a, 'b> {
    const INHERENT_EVIL: fn(&'b u32) -> &'a u32 = { |u| u };
}

// while static methods can *assume* this, we should still
// *check* that it holds at the use site.

fn evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <()>::EVIL(b) // FIXME: should be an error
}

fn indirect_evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <IndirectEvil>::EVIL(b) // FIXME: should be an error
}

fn inherent_evil<'a, 'b>(b: &'b u32) -> &'a u32 {
    <Evil>::INHERENT_EVIL(b)
    //~^ ERROR lifetime may not live long enough
}

fn main() {}

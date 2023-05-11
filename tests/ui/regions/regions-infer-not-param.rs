struct Direct<'a> {
    f: &'a isize
}

struct Indirect1 {
    // Here the lifetime parameter of direct is bound by the fn()
    g: Box<dyn FnOnce(Direct) + 'static>
}

struct Indirect2<'a> {
    // But here it is set to 'a
    g: Box<dyn FnOnce(Direct<'a>) + 'static>
}

fn take_direct<'a,'b>(p: Direct<'a>) -> Direct<'b> { p }
//~^ ERROR lifetime may not live long enough

fn take_indirect1(p: Indirect1) -> Indirect1 { p }

fn take_indirect2<'a,'b>(p: Indirect2<'a>) -> Indirect2<'b> { p }
//~^ ERROR lifetime may not live long enough
//~| ERROR lifetime may not live long enough

fn main() {}

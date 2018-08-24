struct direct<'a> {
    f: &'a isize
}

struct indirect1 {
    // Here the lifetime parameter of direct is bound by the fn()
    g: Box<FnOnce(direct) + 'static>
}

struct indirect2<'a> {
    // But here it is set to 'a
    g: Box<FnOnce(direct<'a>) + 'static>
}

fn take_direct<'a,'b>(p: direct<'a>) -> direct<'b> { p } //~ ERROR mismatched types

fn take_indirect1(p: indirect1) -> indirect1 { p }

fn take_indirect2<'a,'b>(p: indirect2<'a>) -> indirect2<'b> { p } //~ ERROR mismatched types
//~| expected type `indirect2<'b>`
//~| found type `indirect2<'a>`
//~| ERROR mismatched types
//~| expected type `indirect2<'b>`
//~| found type `indirect2<'a>`

fn main() {}

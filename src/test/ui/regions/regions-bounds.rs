// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

// revisions: base nll
// ignore-compare-mode-nll
//[nll] compile-flags: -Z borrowck=mir

struct TupleStruct<'a>(&'a isize);
struct Struct<'a> { x:&'a isize }

fn a_fn1<'a,'b>(e: TupleStruct<'a>) -> TupleStruct<'b> {
    return e;
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn a_fn3<'a,'b>(e: Struct<'a>) -> Struct<'b> {
    return e;
    //[base]~^ ERROR mismatched types
    //[nll]~^^ ERROR lifetime may not live long enough
}

fn main() { }

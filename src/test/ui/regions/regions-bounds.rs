// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct an_enum<'a>(&'a isize);
struct a_class<'a> { x:&'a isize }

fn a_fn1<'a,'b>(e: an_enum<'a>) -> an_enum<'b> {
    return e; //~ ERROR mismatched types
}

fn a_fn3<'a,'b>(e: a_class<'a>) -> a_class<'b> {
    return e; //~ ERROR mismatched types
}

fn main() { }

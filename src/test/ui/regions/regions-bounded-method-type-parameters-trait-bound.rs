// Check that explicit region bounds are allowed on the various
// nominal types (but not on other types) and that they are type
// checked.

struct Inv<'a> { // invariant w/r/t 'a
    x: &'a mut &'a isize
}

trait Foo<'x> {
    fn method<'y:'x>(self, y: Inv<'y>);
}

fn caller1<'a,'b,F:Foo<'a>>(a: Inv<'a>, b: Inv<'b>, f: F) {
    // Here the value provided for 'y is 'a, and hence 'a:'a holds.
    f.method(a);
}

fn caller2<'a,'b,F:Foo<'a>>(a: Inv<'a>, b: Inv<'b>, f: F) {
    // Here the value provided for 'y is 'b, and hence 'b:'a does not hold.
    f.method(b); //~ ERROR lifetime mismatch [E0623]
}

fn caller3<'a,'b:'a,F:Foo<'a>>(a: Inv<'a>, b: Inv<'b>, f: F) {
    // Here the value provided for 'y is 'b, and hence 'b:'a holds.
    f.method(b);
}

fn main() { }

// Test that we can derive lifetime bounds on type parameters
// from trait inheritance.

trait Static : 'static { }

trait Is<'a> : 'a { }

struct Inv<'a> {
    x: Option<&'a mut &'a isize>
}

fn check_bound<'a,A:'a>(x: Inv<'a>, a: A) { }

// In all of these cases, we can derive a bound for A that is longer
// than 'a based on the trait bound of A:

fn foo1<'a,A:Static>(x: Inv<'a>, a: A) {
    check_bound(x, a)
}

fn foo2<'a,A:Static>(x: Inv<'static>, a: A) {
    check_bound(x, a)
}

fn foo3<'a,A:Is<'a>>(x: Inv<'a>, a: A) {
    check_bound(x, a)
}

// In these cases, there is no trait bound, so we cannot derive any
// bound for A and we get an error:

fn bar1<'a,A>(x: Inv<'a>, a: A) {
    check_bound(x, a) //~ ERROR parameter type `A` may not live long enough
}

fn bar2<'a,'b,A:Is<'b>>(x: Inv<'a>, y: Inv<'b>, a: A) {
    check_bound(x, a) //~ ERROR parameter type `A` may not live long enough
}

fn main() { }

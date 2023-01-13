// Test that if a struct declares multiple region bounds for a given
// type parameter, an explicit lifetime bound is required on object
// lifetimes within.

#![allow(dead_code)]

trait Test {
    fn foo(&self) { }
}

struct Ref0<T:?Sized> {
    r: *mut T
}

struct Ref1<'a,T:'a+?Sized> {
    r: &'a T
}

struct Ref2<'a,'b:'a,T:'a+'b+?Sized> {
    r: &'a &'b T
}

fn a<'a,'b>(t: Ref2<'a,'b, dyn Test>) {
    //~^ ERROR lifetime bound for this object type cannot be deduced from context
}

fn b(t: Ref2<dyn Test>) {
    //~^ ERROR lifetime bound for this object type cannot be deduced from context
}

fn c(t: Ref2<&dyn Test>) {
    // In this case, the &'a overrides.
}

fn d(t: Ref2<Ref1<dyn Test>>) {
    // In this case, the lifetime parameter from the Ref1 overrides.
}

fn e(t: Ref2<Ref0<dyn Test>>) {
    // In this case, Ref2 is ambiguous, but Ref0 overrides with 'static.
}

fn f(t: &Ref2<dyn Test>) {
    //~^ ERROR lifetime bound for this object type cannot be deduced from context
}

fn main() {
}

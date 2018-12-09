// normalize-stderr-test: "<\[closure@.+`" -> "$$CLOSURE`"

#![allow(unused)]

#![recursion_limit = "20"]
#![type_length_limit = "20000000"]
#![crate_type = "rlib"]

#[derive(Clone)]
struct A (B);

impl A {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let &A(ref term) = self;
        term.matches(f);
    }
}

#[derive(Clone)]
enum B {
    Variant1,
    Variant2(C),
}

impl B {
    pub fn matches<F: Fn()>(&self, f: &F) {
        match self {
            &B::Variant2(ref factor) => {
                factor.matches(&|| ())
            }
            _ => unreachable!("")
        }
    }
}

#[derive(Clone)]
struct C (D);

impl C {
    pub fn matches<F: Fn()>(&self, f: &F) {
        let &C(ref base) = self;
        base.matches(&|| {
            C(base.clone()).matches(f)
        })
    }
}

#[derive(Clone)]
struct D (Box<A>);

impl D {
    pub fn matches<F: Fn()>(&self, f: &F) {
        //~^ ERROR reached the type-length limit while instantiating `<D>::matches::<[closure
        let &D(ref a) = self;
        a.matches(f)
    }
}

pub fn matches() {
    A(B::Variant1).matches(&(|| ()))
}

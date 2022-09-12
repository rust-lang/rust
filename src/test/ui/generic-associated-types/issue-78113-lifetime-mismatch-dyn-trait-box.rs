// Test for diagnostics when we have mismatched lifetime due to implicit 'static lifetime in GATs

// check-fail

#![feature(generic_associated_types)]

pub trait A {}
impl A for &dyn A {}
impl A for Box<dyn A> {}

pub trait B {
    type T<'a>: A;
}

impl B for () {
    // `'a` doesn't match implicit `'static`: suggest `'_`
    type T<'a> = Box<dyn A + 'a>; //~ incompatible lifetime on type
}

trait C {}
impl C for Box<dyn A + 'static> {}
pub trait D {
    type T<'a>: C;
}
impl D for () {
    // `'a` doesn't match explicit `'static`: we *should* suggest removing `'static`
    type T<'a> = Box<dyn A + 'a>; //~ incompatible lifetime on type
}

trait E {}
impl E for (Box<dyn A>, Box<dyn A>) {}
pub trait F {
    type T<'a>: E;
}
impl F for () {
    // `'a` doesn't match explicit `'static`: suggest `'_`
    type T<'a> = (Box<dyn A + 'a>, Box<dyn A + 'a>); //~ incompatible lifetime on type
}

fn main() {}

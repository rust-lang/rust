#![feature(const_trait_impl)]
#![feature(effects)]
#![feature(inline_const)]

#[const_trait]
trait Foo {
    type AssocTy;
    fn bar();
}

struct A;
impl Foo for A {
    type AssocTy = ();
    fn bar() {
        println!("");
    }
}

struct B;
impl const Foo for B {
    type AssocTy = i32;
    fn bar() {}
}

const C: <A as Foo>::AssocTy = ();
const D: <B as Foo>::AssocTy = 42;

const FOO: () = {
    A::bar(); //~ ERROR: cannot call
    B::bar();
    foo();
    moo::<A>();
    moo::<B>();
    boo::<A>(); //~ ERROR: cannot call
    boo::<B>();
};

static BAR: () = {
    A::bar(); //~ ERROR: cannot call
    B::bar();
    foo();
    moo::<A>();
    moo::<B>();
    boo::<A>(); //~ ERROR: cannot call
    boo::<B>();
};

const fn foo() {
    A::bar(); //~ ERROR: cannot call
    B::bar();
    moo::<A>();
    moo::<B>();
    boo::<A>(); //~ ERROR: cannot call
    boo::<B>();
}

const fn moo<T: Foo>() {
    T::bar(); //~ ERROR: cannot call
}

const fn boo<T: ~const Foo>() {
    T::bar();
}

fn main() {
    A::bar();
    B::bar();
    foo();
    moo::<A>();
    moo::<B>();
    boo::<A>();
    boo::<B>();

    const {
        A::bar(); //~ ERROR: cannot call
        B::bar();
        foo();
        moo::<A>();
        moo::<B>();
        boo::<A>(); //~ ERROR: cannot call
        boo::<B>();
    };

    [(); {
        A::bar(); //~ ERROR: cannot call
        B::bar();
        foo();
        moo::<A>();
        moo::<B>();
        boo::<A>(); //~ ERROR: cannot call
        boo::<B>();
        0
    }];
}

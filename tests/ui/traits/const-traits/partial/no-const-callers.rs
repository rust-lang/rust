#![feature(const_trait_impl, rustc_attrs)]

const trait A {
    fn a();
    #[rustc_non_const_trait_method]
    fn b() { println!("hi"); }
}

const impl A for () {
    fn a() {}
}

const impl A for u8 {
    fn a() {}
    fn b() { println!("hello"); }
    //~^ ERROR: cannot call non-const function
}

const impl A for i8 {
    fn a() {}
    fn b() {}
}

const fn foo<T: [const] A>() {
    T::a();
    T::b();
    //~^ ERROR: cannot call non-const associated function
    <()>::a();
    <()>::b();
    //~^ ERROR: cannot call non-const associated function
    u8::a();
    u8::b();
    //~^ ERROR: cannot call non-const associated function
    i8::a();
    i8::b();
    //~^ ERROR: cannot call non-const associated function
}

fn main() {}

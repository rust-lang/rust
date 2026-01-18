#![feature(const_trait_impl, rustc_attrs)]

const trait A {
    fn a();
    #[rustc_non_const_trait_method]
    fn b() { println!("hi"); }
}

impl const A for () {
    fn a() {}
}

const fn foo<T: [const] A>() {
    T::a();
    T::b();
    //~^ ERROR: cannot call non-const associated function
    <()>::a();
    <()>::b();
    //~^ ERROR: cannot call non-const associated function
}

fn main() {}

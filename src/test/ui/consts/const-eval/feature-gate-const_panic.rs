fn main() {}

const Z: () = panic!("cheese");
//~^ ERROR panicking in constants is unstable

const Y: () = unreachable!();
//~^ ERROR panicking in constants is unstable

const X: () = unimplemented!();
//~^ ERROR panicking in constants is unstable

const fn a() {
    assert!(2 + 2 == 4);
    //~^ ERROR panicking in constant functions is unstable
}

const fn b() {
    panic!("oh no");
    //~^ ERROR panicking in constant functions is unstable
}

const fn c() {
    unimplemented!();
    //~^ ERROR panicking in constant functions is unstable
}

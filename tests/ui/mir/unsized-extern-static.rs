//! Regression test for #129109
//! MIR building used to produce erroneous constants when referring to statics of unsized type.
//@ compile-flags: -Zmir-enable-passes=+GVN -Zvalidate-mir

extern "C" {
    pub static mut symbol: [i8];
    //~^ ERROR the size for values of type `[i8]`
}

fn main() {
    println!("C", unsafe { &symbol });
    //~^ ERROR argument never used
}

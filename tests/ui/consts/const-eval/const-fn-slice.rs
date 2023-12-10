//check-pass

#![feature(const_trait_impl)]
#![feature(fn_traits)]
const fn f() -> usize {
    5
}

const fn main() {
    let _ = [0; Fn::call(&f, ())];
}

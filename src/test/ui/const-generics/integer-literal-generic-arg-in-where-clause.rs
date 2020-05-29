// check-pass

#![feature(const_generics)]
//~^ WARN the feature `const_generics` is incomplete

fn takes_closure_of_array_3<F>(f: F) where F: Fn([i32; 3]) {
    f([1, 2, 3]);
}

fn takes_closure_of_array_3_apit(f: impl Fn([i32; 3])) {
    f([1, 2, 3]);
}

fn returns_closure_of_array_3() -> impl Fn([i32; 3]) {
    |_| {}
}

fn main() {}

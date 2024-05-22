//@ build-fail
//@ compile-flags: -Zmir-opt-level=3

fn foo<T>() {
    if false {
        const { panic!() } //~ ERROR E0080
    }
}

fn main() {
    foo::<i32>();
}

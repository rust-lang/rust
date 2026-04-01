//@ build-fail
//@ compile-flags: -Zmir-opt-level=3
//@ ignore-parallel-frontend post-monomorphization errors
fn foo<T>() {
    if false {
        const { panic!() } //~ ERROR E0080
    }
}

fn main() {
    foo::<i32>();
}

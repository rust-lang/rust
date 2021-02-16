// check-pass

#![allow(unused)]

struct NonCloneType<T>(T);

#[derive(Clone)]
struct CloneType<T>(T);

fn main() {
    let non_clone_type_ref = &NonCloneType(1u32);
    let non_clone_type_ref_clone: &NonCloneType<u32> = non_clone_type_ref.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing

    let clone_type_ref = &CloneType(1u32);
    let clone_type_ref_clone: CloneType<u32> = clone_type_ref.clone();

    // Calling clone on a double reference doesn't warn since the method call itself
    // peels the outer reference off
    let clone_type_ref = &&CloneType(1u32);
    let clone_type_ref_clone: &CloneType<u32> = clone_type_ref.clone();

    let xs = ["a", "b", "c"];
    let _v: Vec<&str> = xs.iter().map(|x| x.clone()).collect(); // ok, but could use `*x` instead
}

fn generic<T>(non_clone_type: &NonCloneType<T>) {
    non_clone_type.clone();
}

fn non_generic(non_clone_type: &NonCloneType<u32>) {
    non_clone_type.clone();
    //~^ WARNING call to `.clone()` on a reference in this situation does nothing
}

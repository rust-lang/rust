//@ compile-flags: -Znext-solver
//@ check-pass

// Make sure that we consider nested obligations when checking whether
// we should coerce fn definitions to function pointers.

fn foo<const N: usize>() {}
fn bar<T>() {}
fn main() {
    let _ = if true { foo::<{ 0 + 0 }> } else { foo::<1> };
    let _ = if true {
        bar::<for<'a> fn(<Vec<&'a ()> as IntoIterator>::Item)>
    } else {
        bar::<fn(i32)>
    };
}

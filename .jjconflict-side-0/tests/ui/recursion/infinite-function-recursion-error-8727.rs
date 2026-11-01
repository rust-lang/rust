// https://github.com/rust-lang/rust/issues/8727
// Verify the compiler fails with an error on infinite function
// recursions.

//@ build-fail
//@ compile-flags: --diagnostic-width=100 -Zwrite-long-types-to-disk=yes

fn generic<T>() { //~ WARN function cannot return without recursing
    generic::<Option<T>>();
}
//~^^ ERROR reached the recursion limit while instantiating `generic::<Option<

fn main () {
    // Use generic<T> at least once to trigger instantiation.
    generic::<i32>();
}

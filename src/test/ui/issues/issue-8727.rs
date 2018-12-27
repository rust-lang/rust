// Verify the compiler fails with an error on infinite function
// recursions.

fn generic<T>() {
    generic::<Option<T>>();
}
//~^^^ ERROR reached the recursion limit while instantiating `generic::<std::option::Option<
//~| WARN function cannot return without recursing



fn main () {
    // Use generic<T> at least once to trigger instantiation.
    generic::<i32>();
}

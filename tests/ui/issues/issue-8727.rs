// Verify the compiler fails with an error on infinite function
// recursions.

//@ build-fail
// The regex below normalizes the long type file name to make it suitable for compare-modes.
//@ normalize-stderr: "'\$TEST_BUILD_DIR/.*\.long-type.txt'" -> "'$$TEST_BUILD_DIR/$$FILE.long-type.txt'"

fn generic<T>() { //~ WARN function cannot return without recursing
    generic::<Option<T>>();
}
//~^^ ERROR reached the recursion limit while instantiating `generic::<Option<


fn main () {
    // Use generic<T> at least once to trigger instantiation.
    generic::<i32>();
}

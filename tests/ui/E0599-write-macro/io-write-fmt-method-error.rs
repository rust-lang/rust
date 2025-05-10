// Issue #139051 - Test for the case where io::Write would be more appropriate
//
// Test that when using write! on a type that doesn't implement std::io::Write trait,
// we get a clear error message suggesting to import the appropriate trait.
//
// edition:2021
// ignore-msvc
// ignore-emscripten
// run-fail
// check-pass

fn main() {
    // Simple struct that doesn't implement std::io::Write
    struct MyIoStruct {
        value: i32,
    }

    let mut s = MyIoStruct { value: 42 };

    // This should generate E0599 with the improved error message
    // suggesting io::Write instead
    write!(s, "Hello, world!"); //~ ERROR cannot write into `MyIoStruct`
}

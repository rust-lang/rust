// Issue #139051 - Ensure we don't get confusing suggestions for E0599
// on write!/writeln! macros
//
// Test that when using write!/writeln! macros with a type that doesn't implement
// std::fmt::Write trait, we get a clear error message without irrelevant suggestions.
//
// edition:2021
// ignore-msvc
// ignore-emscripten
// run-fail
// check-pass

fn main() {
    // Simple struct that doesn't implement std::fmt::Write
    struct MyStruct {
        value: i32,
    }

    let mut s = MyStruct { value: 42 };

    // This should generate E0599 with the improved error message
    // and not suggest irrelevant methods like write_str or push_str
    writeln!(s, "Hello, world!"); //~ ERROR cannot write into `MyStruct`
}

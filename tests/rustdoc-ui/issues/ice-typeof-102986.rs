// https://github.com/rust-lang/rust/issues/102986
struct Struct {
    y: (typeof("hey"),),
    //~^ `typeof` is a reserved keyword but unimplemented
}

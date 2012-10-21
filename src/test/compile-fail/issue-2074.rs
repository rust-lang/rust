// xfail-test
fn main() {
    let one = fn@() -> uint {
        enum r { a };
        return a as uint;
    };
    let two = fn@() -> uint {
        enum r { a };
        return a as uint;
    };
    one(); two();
}

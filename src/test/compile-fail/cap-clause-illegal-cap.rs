// error-pattern: copying a noncopyable value

fn to_lambda2(b: block(uint) -> uint) -> lambda(uint) -> uint {
    // test case where copy clause specifies a value that is not used
    // in lambda body, but value is illegal to copy:
    ret lambda[copy b](u: uint) -> uint { 22u };
}

fn main() {
}

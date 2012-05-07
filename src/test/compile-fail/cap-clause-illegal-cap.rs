// error-pattern: copying a noncopyable value

fn to_lambda2(b: fn(uint) -> uint) -> fn@(uint) -> uint {
    // test case where copy clause specifies a value that is not used
    // in fn@ body, but value is illegal to copy:
    ret fn@(u: uint, copy b) -> uint { 22u };
}

fn main() {
}

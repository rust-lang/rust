// error-pattern: copying a noncopyable value

class foo { let x: int; new(x: int) { self.x = x; } drop { } }

fn to_lambda2(b: foo) -> fn@(uint) -> uint {
    // test case where copy clause specifies a value that is not used
    // in fn@ body, but value is illegal to copy:
    return fn@(u: uint, copy b) -> uint { 22u };
}

fn main() {
}

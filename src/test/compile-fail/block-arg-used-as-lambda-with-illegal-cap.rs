// error-pattern: copying a noncopyable value

fn to_lambda1(f: fn@(uint) -> uint) -> fn@(uint) -> uint {
    ret f;
}

fn to_lambda2(b: block(uint) -> uint) -> fn@(uint) -> uint {
    ret to_lambda1({|x| b(x)});
}

fn main() {
}

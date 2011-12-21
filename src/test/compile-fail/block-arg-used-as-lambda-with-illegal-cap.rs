// error-pattern: copying a noncopyable value

fn to_lambda1(f: lambda(uint) -> uint) -> lambda(uint) -> uint {
    ret f;
}

fn to_lambda2(b: block(uint) -> uint) -> lambda(uint) -> uint {
    ret to_lambda1({|x| b(x)});
}

fn main() {
}

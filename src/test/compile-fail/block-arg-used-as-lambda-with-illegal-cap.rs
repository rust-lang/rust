fn to_lambda1(f: fn@(uint) -> uint) -> fn@(uint) -> uint {
    return f;
}

fn to_lambda2(b: fn(uint) -> uint) -> fn@(uint) -> uint {
    return to_lambda1({|x| b(x)}); //~ ERROR value may contain borrowed pointers
}

fn main() {
}

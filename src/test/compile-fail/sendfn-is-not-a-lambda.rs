// error-pattern: mismatched types: expected `fn@(++uint) -> uint`

fn test(f: fn@(uint) -> uint) -> uint {
    ret f(22u);
}

fn main() {
    let f = sendfn(x: uint) -> uint { ret 4u; };
    log(debug, test(f));
}

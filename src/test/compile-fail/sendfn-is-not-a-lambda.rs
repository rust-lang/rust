// error-pattern: mismatched types: expected 'lambda(++uint) -> uint'

fn test(f: lambda(uint) -> uint) -> uint {
    ret f(22u);
}

fn main() {
    let f = sendfn(x: uint) -> uint { ret 4u; };
    log(debug, test(f));
}

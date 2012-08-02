fn test(f: fn@(uint) -> uint) -> uint {
    return f(22u);
}

fn main() {
    let f = fn~(x: uint) -> uint { return 4u; };
    log(debug, test(f)); //~ ERROR expected `fn@(uint) -> uint`
}

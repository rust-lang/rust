fn test(f: fn@(uint) -> uint) -> uint {
    ret f(22u);
}

fn main() {
    let f = fn~(x: uint) -> uint { ret 4u; };
    log(debug, test(f)); //! ERROR expected `fn@(uint) -> uint`
}

struct X {}

const Y: X = X("ö"); //~ ERROR expected function, found struct `X`

fn main() {}

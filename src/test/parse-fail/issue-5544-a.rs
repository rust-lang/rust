// compile-flags: -Z parse-only

fn main() {
    let __isize = 340282366920938463463374607431768211456; // 2^128
    //~^ ERROR int literal is too large
}

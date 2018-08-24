struct S(usize, usize, usize, usize);

fn main() {
    if let S { a, b, c, d } = S(1, 2, 3, 4) {
    //~^ ERROR struct `S` does not have fields named `a`, `b`, `c`, `d` [E0026]
    //~| ERROR pattern does not mention fields `0`, `1`, `2`, `3` [E0027]
        println!("hi");
    }
}

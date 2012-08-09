fn main() {
    let a: ~[int] = ~[];
    vec::each(a, fn@(_x: int) -> bool { //~ ERROR not all control paths return a value
    });
}

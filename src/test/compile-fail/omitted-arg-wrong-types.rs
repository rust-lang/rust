fn let_in<T>(x: T, f: fn(T)) {}

fn main() {
    let_in(3u, fn&(i) { assert i == 3; });
    //!^ ERROR expected `uint` but found `int`

    let_in(3, fn&(i) { assert i == 3u; });
    //!^ ERROR expected `int` but found `uint`
}
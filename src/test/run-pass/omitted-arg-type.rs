fn let_in<T>(x: T, f: fn(T)) {}

fn main() {
    let_in(3u) { |i| assert i == 3u; };
    let_in(3) { |i| assert i == 3; };
    let_in(3u, fn&(i) { assert i == 3u; });
    let_in(3, fn&(i) { assert i == 3; });
}
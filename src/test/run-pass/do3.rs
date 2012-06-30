fn f(f: fn@(int) -> int) -> int { f(10) }

fn main() {
    assert do f |i| { i } == 10;
}

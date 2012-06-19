fn f(f: fn@(int) -> bool) -> bool { f(10) }

fn main() {
    assert do f() { |i| i == 10 } == 10; //! ERROR: expected `bool` but found `int`
}

fn f(f: fn@(int) -> bool) -> bool { f(10i) }

fn main() {
    assert do f() |i| { i == 10i } == 10i;
    //~^ ERROR: expected `bool` but found `int`
}

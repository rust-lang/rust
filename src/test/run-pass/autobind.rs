fn f<T: Copy>(x: ~[T]) -> T { return x[0]; }

fn g(act: fn(~[int]) -> int) -> int { return act(~[1, 2, 3]); }

fn main() {
    assert (g(f) == 1);
    let f1: fn(~[~str]) -> ~str = f;
    assert (f1(~[~"x", ~"y", ~"z"]) == ~"x");
}

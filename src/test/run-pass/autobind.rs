// FIXME: Autobind doesn't work for bare functions currently
// because it would have to convert them to shared closures
// xfail-test

fn f<T>(x: [T]) -> T { ret x[0]; }

fn g(act: fn([int]) -> int) -> int { ret act([1, 2, 3]); }

fn main() {
    assert (g(f) == 1);
    let f1: fn([str]) -> str = f;
    assert (f1(["x", "y", "z"]) == "x");
}

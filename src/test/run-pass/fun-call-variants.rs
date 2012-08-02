// -*- rust -*-
fn ho(f: fn@(int) -> int) -> int { let n: int = f(3); return n; }

fn direct(x: int) -> int { return x + 1; }

fn main() {
    let a: int = direct(3); // direct
    let b: int = ho(direct); // indirect unbound

    assert (a == b);
}

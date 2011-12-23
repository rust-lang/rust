pure fn p(j: int) -> bool { true }

fn f(i: int, j: int) : p(j) -> int { j }

fn g(i: int, j: int) : p(j) -> int { f(i, j) }

fn main() { let x = 1; check (p(x)); log(debug, g(x, x)); }

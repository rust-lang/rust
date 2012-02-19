fn foo(i: int) -> int { i + 1 }

fn apply<A>(f: fn(A) -> A, v: A) -> A { f(v) }

fn main() {
    let f = {|i| foo(i)};
    assert apply(f, 2) == 3;
}

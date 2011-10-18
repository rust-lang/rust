fn fix_help<A, ~B>(f: fn(fn@(A) -> B, A) -> B, x: A) -> B {
    ret f(bind fix_help(f, _), x);
}

fn fix<A, ~B>(f: fn(fn@(A) -> B, A) -> B) -> fn@(A) -> B {
    ret bind fix_help(f, _);
}

fn fact_(f: fn@(&&int) -> int, &&n: int) -> int {
    // fun fact 0 = 1
    ret if n == 0 { 1 } else { n * f(n - 1) };
}

fn main() {
    let fact = fix(fact_);
    assert (fact(5) == 120);
    assert (fact(2) == 2);
}

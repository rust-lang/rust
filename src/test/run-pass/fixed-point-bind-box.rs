// xfail-fast
#[legacy_modes];

fn fix_help<A, B>(f: extern fn(fn@(A) -> B, A) -> B, x: A) -> B {
    return f({|a|fix_help(f, a)}, x);
}

fn fix<A, B>(f: extern fn(fn@(A) -> B, A) -> B) -> fn@(A) -> B {
    return {|a|fix_help(f, a)};
}

fn fact_(f: fn@(&&v: int) -> int, &&n: int) -> int {
    // fun fact 0 = 1
    return if n == 0 { 1 } else { n * f(n - 1) };
}

fn main() {
    let fact = fix(fact_);
    assert (fact(5) == 120);
    assert (fact(2) == 2);
}

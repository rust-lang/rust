// xfail-stage0

fn fix_help[A,B](@fn (@fn (&A) -> B, &A) -> B f, &A x) -> B {
    ret f(@bind fix_help(f, _), x);
}

fn fix[A,B](@fn (@fn (&A) -> B, &A) -> B f) -> (@fn(&A) -> B) {
    ret @bind fix_help(f, _);
}

fn fact_(@fn (&int) -> int f, &int n) -> int {
    // fun fact 0 = 1
    ret if (n == 0) { 1 } else { n*f(n-1) };
}

fn main() {
    auto fact = fix(@fact_);
    assert(fact(5) == 120);
    assert(fact(2) == 2);
}

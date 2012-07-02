// Testing that we can drop the || in for/do exprs

fn f(f: fn@() -> bool) { }

fn d(f: fn@()) { }

fn main() {
    for f { }
    do d { }
}
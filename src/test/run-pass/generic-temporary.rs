

fn mk() -> int { ret 1; }

fn chk(&&a: int) { log(debug, a); assert (a == 1); }

fn apply<T>(produce: native fn() -> T,
            consume: native fn(T)) {
    consume(produce());
}

fn main() {
    let produce: native fn() -> int = mk;
    let consume: native fn(&&int) = chk;
    apply::<int>(produce, consume);
}

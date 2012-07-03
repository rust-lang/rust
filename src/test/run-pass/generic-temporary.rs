

fn mk() -> int { ret 1; }

fn chk(&&a: int) { log(debug, a); assert (a == 1); }

fn apply<T>(produce: extern fn() -> T,
            consume: extern fn(T)) {
    consume(produce());
}

fn main() {
    let produce: extern fn() -> int = mk;
    let consume: extern fn(&&int) = chk;
    apply::<int>(produce, consume);
}

// Make sure we don't leak fn@s in silly ways.
fn force(f: fn@()) { f() }
fn main() {
    let x = 7;
    let _f = fn@() { log(error, x); };
    force(fn@() { log(error, x); });
}

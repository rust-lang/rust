// Make sure we don't leak lambdas in silly ways.
fn force(f: fn@()) { f() }
fn main() {
    let x = 7;
    lambda () { log_full(core::error, x); };
    force(lambda () { log_full(core::error, x); });
}

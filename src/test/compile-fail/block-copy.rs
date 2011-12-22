// error-pattern: block type can only appear

fn lol(f: block()) -> block() { ret f; }
fn main() {
    let i = 8;
    let f = lol(block () { log_full(core::error, i); });
    f();
}

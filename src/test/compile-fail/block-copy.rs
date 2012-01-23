// error-pattern: block type can only appear

fn lol(f: fn()) -> fn() { ret f; }
fn main() {
    let i = 8;
    let f = lol(fn&() { log(error, i); });
    f();
}

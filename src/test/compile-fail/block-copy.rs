// error-pattern: non-copyable
// xfail-stage0

fn lol(&block() f) -> block() { ret f; }
fn main() {
    auto i = 8;
    auto f = lol(block() { log_err i; } );
    f();
}

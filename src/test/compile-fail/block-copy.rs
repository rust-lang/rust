// error-pattern: non-copyable
// xfail-stage0

fn lol(f: &block() ) -> block()  { ret f; }
fn main() { let i = 8; let f = lol(block () { log_err i; }); f(); }
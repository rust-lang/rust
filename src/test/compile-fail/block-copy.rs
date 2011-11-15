// error-pattern: needed shared type, got pinned type block
// xfail-test

fn lol(f: block()) -> block() { ret f; }
fn main() { let i = 8; let f = lol(block () { log_err i; }); f(); }

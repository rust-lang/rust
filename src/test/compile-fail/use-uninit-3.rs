// error-pattern:Unsatisfied precondition

fn foo(x: int) { log x; }

fn main() { let x: int; if 1 > 2 { log "whoops"; } else { x = 10; } foo(x); }

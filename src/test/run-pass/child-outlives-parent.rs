// Reported as issue #126, child leaks the string.

extern mod std;

fn child2(&&s: ~str) { }

fn main() { let x = task::spawn(|| child2(~"hi") ); }

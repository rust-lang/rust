// error-pattern:too many arguments

extern mod std;

fn main() { let s = fmt!("%s", "test", "test"); }

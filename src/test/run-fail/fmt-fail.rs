// error-pattern:meh
extern mod std;

fn main() { let str_var: ~str = ~"meh"; fail fmt!("%s", str_var); }

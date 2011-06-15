

fn f(@int x) { }

fn main() { auto x = @10; auto ff = bind f(_); ff(x); ff(x); }
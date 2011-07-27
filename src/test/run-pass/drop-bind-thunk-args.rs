

fn f(x: @int) { }

fn main() { let x = @10; let ff = bind f(_); ff(x); ff(x); }
type quux = {bar: int};

fn g(i: &int) { }
fn f(foo: @@quux) { g(foo.bar); }

fn main() { }
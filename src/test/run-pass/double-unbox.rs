type quux = rec(int bar);

fn g(&int i) { }
fn f(@@quux foo) {
    g(foo.bar);
}

fn main() {}

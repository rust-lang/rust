//@ known-bug: #134162

fn main() {
    struct X;

    let xs = [X, X, X];
    let eq = xs == [panic!("panic evaluated"); 2];
}

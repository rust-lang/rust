// error-pattern: mismatched types

fn main() {
    let v = [mutable @mutable ~mutable [0]];

    fn f(&&v: [mutable @mutable ~mutable [const int]]) {
    }

    f(v);
}

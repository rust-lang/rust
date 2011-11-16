// error-pattern: mismatched types

fn main() {
    let v = ~mutable [0];

    fn f(&&v: ~mutable [const int]) {
        *v = [mutable 3]
    }

    f(v);
}

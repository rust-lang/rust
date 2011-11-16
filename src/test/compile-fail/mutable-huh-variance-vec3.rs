// error-pattern: mismatched types

fn main() {
    let v = [mutable [mutable [0]]];

    fn f(&&v: [mutable [mutable [const int]]]) {
        v[0][1] = [mutable 3]
    }

    f(v);
}

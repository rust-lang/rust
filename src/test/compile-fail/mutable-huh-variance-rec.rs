// error-pattern: mismatched types

fn main() {
    let v = {mutable g: [0]};

    fn f(&&v: {mutable g: [const int]}) {
        v.g = [mutable 3]
    }

    f(v);
}

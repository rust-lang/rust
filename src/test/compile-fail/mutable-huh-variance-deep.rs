// error-pattern: mismatched types

fn main() {
    let v = [mut @mut ~mut [0]];

    fn f(&&v: [mut @mut ~mut [const int]]) {
    }

    f(v);
}

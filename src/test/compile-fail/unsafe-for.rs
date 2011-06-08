// error-pattern:invalidate alias x

fn main() {
    let vec[mutable int] v = [mutable 1, 2, 3];
    for (int x in v) {
        v.(0) = 10;
        log x;
    }
}

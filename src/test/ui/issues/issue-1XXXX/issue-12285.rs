// run-pass

struct S;

fn main() {
    match Some(&S) {
        Some(&S) => {},
        _x => unreachable!()
    }
    match Some(&S) {
        Some(&S) => {},
        None => unreachable!()
    }
}

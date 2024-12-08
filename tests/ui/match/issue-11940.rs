//@ run-pass

const TEST_STR: &'static str = "abcd";

fn main() {
    let s = "abcd";
    match s {
        TEST_STR => (),
        _ => unreachable!()
    }
}

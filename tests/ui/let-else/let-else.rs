//@ run-pass

fn main() {
    let Some(x) = Some(1) else {
        return;
    };
    assert_eq!(x, 1);
}

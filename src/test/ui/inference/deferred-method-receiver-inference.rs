// check-pass

fn foo() -> Vec<u16> {
    let mut s = vec![].into_iter().collect();
    s.push(0);
    s
}

fn main() {
    let _ = foo();
}

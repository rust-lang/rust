fn foo(x: str/&) -> u8 {
    x[0]
}

fn main() {
    let p = "hello"/~;
    let r = foo(p);
    assert r == 'h' as u8;
}

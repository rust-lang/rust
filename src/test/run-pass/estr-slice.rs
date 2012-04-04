// xfail-test
fn main() {
    let x : str/& = "hello";
    let mut y = "there";
    y = x;
    assert y[1] == 'h' as u8;
    assert y[4] == 'e' as u8;
}
fn main() {
    let x = "hello"/&;
    let mut y = "there"/&;
    y = x;
    assert y[0] == 'h' as u8;
    assert y[4] == 'o' as u8;
}
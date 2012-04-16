fn main() {
    let x : str/~ = "hello"/~;
    let _y : str/~ = "there"/~;
    let mut z = "thing"/~;
    z = x;
    assert z[0] == ('h' as u8);
    assert z[4] == ('o' as u8);
}

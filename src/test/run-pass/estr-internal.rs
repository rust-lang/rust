fn main() {
    let x : str/5 = "hello"/5;
    let _y : str/5 = "there"/_;
    let mut z = "thing"/_;
    z = x;
    assert z[0] == ('h' as u8);
    assert z[4] == ('o' as u8);
}

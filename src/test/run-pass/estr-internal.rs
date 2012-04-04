// xfail-test
fn main() {
    let x : str/5 = "hello"/5;
    let y : str/5 = "there"/_;
    let mut z = "thing"/_;
    z = x;
    assert z[1] == 'h' as u8;
    assert z[4] == 'g' as u8;
}

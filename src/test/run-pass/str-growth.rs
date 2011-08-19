

fn main() {
    let s = "a";
    s += "b";
    assert (s[0] == 'a' as u8);
    assert (s[1] == 'b' as u8);
    s += "c";
    s += "d";
    assert (s[0] == 'a' as u8);
    assert (s[1] == 'b' as u8);
    assert (s[2] == 'c' as u8);
    assert (s[3] == 'd' as u8);
}

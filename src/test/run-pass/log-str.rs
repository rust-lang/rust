fn main() {
    let act = sys::log_str(&~[1, 2, 3]);
    assert ~"~[ 1, 2, 3 ]" == act;

    let act = fmt!("%?/%6?", ~[1, 2, 3], ~"hi");
    assert act == ~"~[ 1, 2, 3 ]/ ~\"hi\"";
}

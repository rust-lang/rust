// Check that issue #954 stays fixed

fn main() {
    match -1 { -1 => {}, _ => fail ~"wat" }
    assert 1-1 == 0;
}

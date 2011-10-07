// Check that issue #954 stays fixed

fn main() {
    alt -1 { -1 {} }
    assert 1-1 == 0;
}

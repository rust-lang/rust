

fn main() {
    let char c = 'x';
    let char d = 'x';
    assert (c == 'x');
    assert ('x' == c);
    assert (c == c);
    assert (c == d);
    assert (d == c);
    assert (d == 'x');
    assert ('x' == d);
}


fn main() {
    let c: char = 'x';
    let d: char = 'x';
    assert (c == 'x');
    assert ('x' == c);
    assert (c == c);
    assert (c == d);
    assert (d == c);
    assert (d == 'x');
    assert ('x' == d);
}
import swappable::{swappable, methods};

fn main() {
    let d = swappable(3);
    assert d.get() == 3;
    d.set(4);
    assert d.get() == 4;
    d.swap { |i| i + 1 };
    assert d.get() == 5;
    assert d.with { |i| i + 1 } == 6;
    assert d.get() == 5;
    assert swappable::unwrap(d) == 5;
}
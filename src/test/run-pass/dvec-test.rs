import dvec::{dvec, extensions};

fn main() {
    let d = dvec();
    d.push(3);
    d.push(4);
    assert d.get() == [3, 4];
    d.set([mut 5]);
    d.push(6);
    d.push(7);
    d.push(8);
    d.push(9);
    d.push(10);
    d.push_vec([11, 12, 13]);

    let exp = [5, 6, 7, 8, 9, 10, 11, 12, 13];
    assert d.get() == exp;
    assert d.get() == exp;
    assert d.len() == exp.len();

    for d.eachi { |i, e|
        assert e == exp[i];
    }

    assert dvec::unwrap(d) == [5, 6, 7, 8, 9, 10, 11, 12, 13];
}
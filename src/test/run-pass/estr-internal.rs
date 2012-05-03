fn main() {
    let x : str/5 = "hello"/5;
    let _y : str/5 = "there"/_;
    let mut z = "thing"/_;
    z = x;
    assert z[0] == ('h' as u8);
    assert z[4] == ('o' as u8);

    let a = "aaaa"/_;
    let b = "bbbb"/_;
    let c = "cccc"/_;

    log(debug, a);

    assert a < b;
    assert a <= b;
    assert a != b;
    assert b >= a;
    assert b > a;

    log(debug, b);

    assert b < c;
    assert b <= c;
    assert b != c;
    assert c >= b;
    assert c > b;

    assert a < c;
    assert a <= c;
    assert a != c;
    assert c >= a;
    assert c > a;

    log(debug, c);
}


fn main() {
    let x = &"hello";
    let v = &"hello";
    let mut y : &str = &"there";

    log(debug, x);
    log(debug, y);

    assert x[0] == 'h' as u8;
    assert x[4] == 'o' as u8;

    let z : &str = &"thing";
    assert v == x;
    assert x != z;

    let a = &"aaaa";
    let b = &"bbbb";

    // let c = &"cccc";
    // let cc = &"ccccc";

    log(debug, a);

    assert a < b;
    assert a <= b;
    assert a != b;
    assert b >= a;
    assert b > a;

    log(debug, b);

// FIXME #3138: So then, why don't these ones work?

/*
    assert a < c;
    assert a <= c;
    assert a != c;
    assert c >= a;
    assert c > a;

    log(debug, c);

    assert c < cc;
    assert c <= cc;
    assert c != cc;
    assert cc >= c;
    assert cc > c;

    log(debug, cc);
*/
}
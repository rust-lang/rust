fn main() {
    let x : &[int] = &[1,2,3,4,5];
    let mut z = &[1,2,3,4,5];
    z = x;
    assert z[0] == 1;
    assert z[4] == 5;

    let a : &[int] = &[1,1,1,1,1];
    let b : &[int] = &[2,2,2,2,2];
    let c : &[int] = &[2,2,2,2,3];
    let cc : &[int] = &[2,2,2,2,2,2];

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

    assert a < cc;
    assert a <= cc;
    assert a != cc;
    assert cc >= a;
    assert cc > a;

    log(debug, cc);
}

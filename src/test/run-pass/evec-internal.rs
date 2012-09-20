// xfail-test
// xfail-fast

// Doesn't work; needs a design decision.

fn main() {
    let x : [int]/5 = [1,2,3,4,5]/5;
    let _y : [int]/5 = [1,2,3,4,5]/_;
    let mut z = [1,2,3,4,5]/_;
    z = x;
    assert z[0] == 1;
    assert z[4] == 5;

    let a : [int]/5 = [1,1,1,1,1]/_;
    let b : [int]/5 = [2,2,2,2,2]/_;
    let c : [int]/5 = [2,2,2,2,3]/_;

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

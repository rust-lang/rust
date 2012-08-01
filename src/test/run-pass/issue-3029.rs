// xfail-test
fn fail_then_concat() {
    let x = ~[], y = ~[3];
    fail;
    x += y;
    ~"good" + ~"bye";
}


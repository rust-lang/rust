// run-pass
fn main() {
    let v1 = { 1 + {2} * {3} };
    let v2 =   1 + {2} * {3}  ;

    assert_eq!(7, v1);
    assert_eq!(7, v2);

    let v3;
    v3 = { 1 + {2} * {3} };
    let v4;
    v4 = 1 + {2} * {3};
    assert_eq!(7, v3);
    assert_eq!(7, v4);

    let v5 = { 1 + {2} * 3 };
    assert_eq!(7, v5);

    let v9 = { 1 + if 1 > 2 {1} else {2} * {3} };
    assert_eq!(7, v9);
}

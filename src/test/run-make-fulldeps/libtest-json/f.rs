#[test]
fn a() {
    println!("print from successful test");
    // Should pass
}

#[test]
fn b() {
    assert!(false);
}

#[test]
#[should_panic]
fn c() {
    assert!(false);
}

#[test]
#[ignore]
fn d() {
    assert!(false);
}

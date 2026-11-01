#[test]
fn a() {
    println!("print from successful test");
    // Should pass
}

#[test]
fn b() {
    println!("print from failing test");
    assert!(false);
}

#[test]
#[should_panic]
fn c() {
    assert!(false);
}

#[test]
#[ignore = "msg"]
fn d() {
    assert!(false);
}

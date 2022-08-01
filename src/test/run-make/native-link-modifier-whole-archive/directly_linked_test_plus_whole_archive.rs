use std::io::Write;

#[test]
fn test_thing() {
    print!("ran the test");
    std::io::stdout().flush().unwrap();
}

#[test]
fn test_stack_assign() {
    let s: String = "a".to_string();
    println!("{}", s.clone());
    let t: String = "a".to_string();
    assert_eq!(s, t);
    let u: String = "b".to_string();
    assert!(s != u);
}

#[test]
fn test_heap_lit() {
    "a big string".to_string();
}

#[test]
fn test_heap_assign() {
    let s: String = "a big ol' string".to_string();
    let t: String = "a big ol' string".to_string();
    assert_eq!(s, t);
    let u: String = "a bad ol' string".to_string();
    assert!(s != u);
}

#[test]
fn test_heap_log() {
    let s = "a big ol' string".to_string();
    println!("{}", s);
}

#[test]
fn test_append() {
    let mut s = String::new();
    s.push_str("a");
    assert_eq!(s, "a");

    let mut s = String::from("a");
    s.push_str("b");
    println!("{}", s.clone());
    assert_eq!(s, "ab");

    let mut s = String::from("c");
    s.push_str("offee");
    assert_eq!(s, "coffee");

    s.push_str("&tea");
    assert_eq!(s, "coffee&tea");
}

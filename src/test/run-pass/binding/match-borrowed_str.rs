// run-pass

fn f1(ref_string: &str) -> String {
    match ref_string {
        "a" => "found a".to_string(),
        "b" => "found b".to_string(),
        _ => "not found".to_string()
    }
}

fn f2(ref_string: &str) -> String {
    match ref_string {
        "a" => "found a".to_string(),
        "b" => "found b".to_string(),
        s => format!("not found ({})", s)
    }
}

fn g1(ref_1: &str, ref_2: &str) -> String {
    match (ref_1, ref_2) {
        ("a", "b") => "found a,b".to_string(),
        ("b", "c") => "found b,c".to_string(),
        _ => "not found".to_string()
    }
}

fn g2(ref_1: &str, ref_2: &str) -> String {
    match (ref_1, ref_2) {
        ("a", "b") => "found a,b".to_string(),
        ("b", "c") => "found b,c".to_string(),
        (s1, s2) => format!("not found ({}, {})", s1, s2)
    }
}

pub fn main() {
    assert_eq!(f1("b"), "found b".to_string());
    assert_eq!(f1("c"), "not found".to_string());
    assert_eq!(f1("d"), "not found".to_string());
    assert_eq!(f2("b"), "found b".to_string());
    assert_eq!(f2("c"), "not found (c)".to_string());
    assert_eq!(f2("d"), "not found (d)".to_string());
    assert_eq!(g1("b", "c"), "found b,c".to_string());
    assert_eq!(g1("c", "d"), "not found".to_string());
    assert_eq!(g1("d", "e"), "not found".to_string());
    assert_eq!(g2("b", "c"), "found b,c".to_string());
    assert_eq!(g2("c", "d"), "not found (c, d)".to_string());
    assert_eq!(g2("d", "e"), "not found (d, e)".to_string());
}

// Test old and new syntax for inclusive range patterns.

fn main() {
    assert!(match 42 { 0 ... 100 => true, _ => false });
    assert!(match 42 { 0 ..= 100 => true, _ => false });

    assert!(match 'x' { 'a' ... 'z' => true, _ => false });
    assert!(match 'x' { 'a' ..= 'z' => true, _ => false });
}


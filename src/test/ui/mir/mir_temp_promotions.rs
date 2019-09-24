// run-pass
fn test1(f: f32) -> bool {
    // test that we properly promote temporaries to allocas when a temporary is assigned to
    // multiple times (assignment is still happening once ∀ possible dataflows).
    !(f.is_nan() || f.is_infinite())
}

fn main() {
    assert_eq!(test1(0.0), true);
}

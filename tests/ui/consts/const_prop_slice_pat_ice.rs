//@ check-pass

fn main() {
    match &[0, 1] as &[i32] {
        [a @ .., x] => {}
        &[] => {}
    }
}

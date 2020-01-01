// check-pass
#![feature(slice_patterns)]

fn main() {
    match &[0, 1] as &[i32] {
        [a @ .., x] => {}
        &[] => {}
    }
}

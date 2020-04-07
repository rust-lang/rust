static mut LEAKER: Option<Box<Vec<i32>>> = None;

fn main() {
    // Having memory "leaked" in globals is allowed.
    unsafe {
        LEAKER = Some(Box::new(vec![0; 42]));
    }
}

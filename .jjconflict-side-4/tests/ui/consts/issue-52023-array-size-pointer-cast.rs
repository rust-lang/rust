fn main() {
    let _ = [0; (&0 as *const i32) as usize]; //~ ERROR pointers cannot be cast to integers during const eval
}

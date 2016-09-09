static mut X: usize = 5;

fn main() {
    unsafe {
        X = 6;
        assert_eq!(X, 6);
    }
}

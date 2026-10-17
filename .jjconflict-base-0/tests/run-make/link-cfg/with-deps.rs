extern crate dep;

fn main() {
    unsafe {
        let v = dep::my_function();
        if cfg!(foo) {
            assert_eq!(v, 1);
        } else if cfg!(bar) {
            assert_eq!(v, 2);
        } else {
            panic!("unknown");
        }
    }
}

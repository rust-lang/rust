extern crate dep_with_staticlib;

fn main() {
    unsafe {
        let v = dep_with_staticlib::my_function();
        if cfg!(foo) {
            assert_eq!(v, 1);
        } else if cfg!(bar) {
            assert_eq!(v, 3);
        } else {
            panic!("unknown");
        }
    }
}

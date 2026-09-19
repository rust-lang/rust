#[link(name = "helper", kind = "static")]
extern "C" {
    fn helper_function() -> i32;
}

fn main() {
    unsafe {
        assert!(42 == helper_function());
    }
}

#![feature(link_cfg)]

#[link(name = "return1", cfg(foo))]
#[link(name = "return2", cfg(bar))]
extern {
    fn my_function() -> i32;
}

fn main() {
    unsafe {
        let v = my_function();
        if cfg!(foo) {
            assert_eq!(v, 1);
        } else if cfg!(bar) {
            assert_eq!(v, 2);
        } else {
            panic!("unknown");
        }
    }
}

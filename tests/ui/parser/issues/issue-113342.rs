#[link(name = "my_c_library")]
extern "C" {
    fn my_c_function(x: i32) -> bool;
}

#[no_mangle]
extern "C" pub fn id(x: i32) -> i32 { x } //~ ERROR expected `fn`, found keyword `pub`

fn main() {}

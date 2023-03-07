#[no_mangle]
pub fn cold_function(c: u8) {
    println!("cold {}", c);
}

#[no_mangle]
pub fn hot_function(c: u8) {
    std::env::set_var(format!("var{}", c), format!("hot {}", c));
}

fn main() {
    let arg = std::env::args().skip(1).next().unwrap();

    for i in 0 .. 1000_000 {
        let some_value = arg.as_bytes()[i % arg.len()];
        if some_value == b'!' {
            // This branch is never taken at runtime
            cold_function(some_value);
        } else {
            hot_function(some_value);
        }
    }
}

// This file has unexpected closing delimiter,

fn func(o: Option<u32>) {
    match o {
        Some(_x) => {}   // Extra '}'
            let _ = if true {};
        }
        None => {}
    }
} //~ ERROR unexpected closing delimiter

fn main() {}

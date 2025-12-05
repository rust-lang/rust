// This file has unexpected closing delimiter,

fn func(o: Option<u32>) {
    match o {
        Some(_x) =>   // Missing '{'
            let _ = if true {};
        }
        None => {}
    }
} //~ ERROR unexpected closing delimiter

fn main() {}

fn my_err(s: String) -> ! { println!("{}", s); panic!(); }

fn ok(i: usize) -> isize {
    if i == 3 {
        my_err("I don't like three".to_string());
    } else {
        return 42;
    }
}

pub fn main() { ok(4); }

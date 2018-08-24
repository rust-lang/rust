fn decode() -> String {
    'outer: loop {
        let mut ch_start: usize;
        break 'outer;
    }
    "".to_string()
}

pub fn main() {
    println!("{}", decode());
}

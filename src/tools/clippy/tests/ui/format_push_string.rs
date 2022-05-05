#![warn(clippy::format_push_string)]

fn main() {
    let mut string = String::new();
    string += &format!("{:?}", 1234);
    string.push_str(&format!("{:?}", 5678));
}

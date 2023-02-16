#![warn(clippy::unnecessary_literal_unwrap)]

fn main() {
    let val = Some(1);
    let _val2 = val.unwrap();
}

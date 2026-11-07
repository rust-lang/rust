#[doc = include_str!("input.md")]
pub struct SomeStruct;

pub fn main() {
    const INPUT_TXT: &'static str = include_str!("input.txt");
    const INPUT_BIN: &'static [u8] = include_bytes!("input.bin");

    println!("{}", INPUT_TXT);
    println!("{:?}", INPUT_BIN);
}

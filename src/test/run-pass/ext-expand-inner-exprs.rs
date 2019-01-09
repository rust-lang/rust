static FOO : &'static str = concat!(concat!("hel", "lo"), "world");

pub fn main() {
    assert_eq!(FOO, "helloworld");
}

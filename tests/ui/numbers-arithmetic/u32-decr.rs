//@ run-pass




pub fn main() {
    let mut word: u32 = 200000;
    word = word - 1;
    assert_eq!(word, 199999);
}

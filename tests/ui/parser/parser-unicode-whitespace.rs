//@ run-pass
// Beware editing: it has numerous whitespace characters which are important.
// It contains one ranges from the 'PATTERN_WHITE_SPACE' property outlined in
// https://unicode.org/Public/UNIDATA/PropList.txt
//
// The characters in the first expression of the assertion can be generated
// from: "4\u{0C}+\n\t\r7\t*\u{20}2\u{85}/\u{200E}3\u{200F}*\u{2028}2\u{2029}"
pub fn main() {
assert_eq!(4+

7   * 2/‎3‏* 2 , 4 + 7 * 2 / 3 * 2);
}

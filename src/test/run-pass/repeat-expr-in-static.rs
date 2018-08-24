static FOO: [isize; 4] = [32; 4];
static BAR: [isize; 4] = [32, 32, 32, 32];

pub fn main() {
    assert_eq!(FOO, BAR);
}

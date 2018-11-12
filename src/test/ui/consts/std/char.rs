// run-pass

static X: bool = 'a'.is_ascii();
static Y: bool = 'ä'.is_ascii();

fn main() {
    assert!(X);
    assert!(!Y);
}

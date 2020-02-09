// run-pass

static X: bool = 'a'.is_ascii();
static Y: bool = 'ä'.is_ascii();

static BX: bool = b'a'.is_ascii();
static BY: bool = 192u8.is_ascii();

fn main() {
    assert!(X);
    assert!(!Y);

    assert!(BX);
    assert!(!BY);
}

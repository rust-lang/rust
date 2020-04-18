// run-pass
// compile-flags: -Zunleash-the-miri-inside-of-you

static DATA: [u8; 3] = *b"abc";

const REF: &'static [u8] = &DATA;
//~^ WARN skipping const checks

const PTR: *const u8 = DATA.as_ptr();
//~^ WARN skipping const checks

fn parse_slice(r: &[u8]) -> bool {
    match r {
        REF => true,
        _ => false,
    }
}

fn parse_id(p: *const u8) -> bool {
    match p {
        p if p == PTR => true,
        _ => false,
    }
}

fn main() {
    assert!(parse_slice(&DATA));
    assert!(parse_id(DATA.as_ptr()));
}

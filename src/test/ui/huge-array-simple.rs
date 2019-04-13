// error-pattern: too big for the current architecture

// normalize-stderr-test "; \d+]" -> "; N]"

// FIXME https://github.com/rust-lang/rust/issues/59774
// normalize-stderr-test "thread.*panicked.*Metadata module not compiled.*\n" -> ""
// normalize-stderr-test "note:.*RUST_BACKTRACE=1.*\n" -> ""
#![allow(exceeding_bitshifts)]

#[cfg(target_pointer_width = "64")]
fn main() {
    let _fat : [u8; (1<<61)+(1<<31)] =
        [0; (1u64<<61) as usize +(1u64<<31) as usize];
}

#[cfg(target_pointer_width = "32")]
fn main() {
    let _fat : [u8; (1<<31)+(1<<15)] =
        [0; (1u32<<31) as usize +(1u32<<15) as usize];
}

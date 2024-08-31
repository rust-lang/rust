//@ known-bug: rust-lang/rust#129503

use std::arch::asm;

unsafe fn f6() {
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "r} {}"));
}

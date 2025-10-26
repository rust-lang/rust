//@ known-bug: #131292
//@ needs-asm-support
use std::arch::asm;

unsafe fn f6() {
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "{}/day{:02}.txt"));
}

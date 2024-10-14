//@ known-bug: #131292
//@ only-x86_64
use std::arch::asm;

unsafe fn f6() {
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "{}/day{:02}.txt"));
}

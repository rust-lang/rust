// Regression test for ICEs #129503 and #131292
//
// Tests that we come up with decent error spans
// when the template fed to `asm!()` is itself a
// macro call like `concat!()` and should not ICE

//@ needs-asm-support

use std::arch::asm;

fn main() {
    // Should not ICE (test case for #129503)
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "r} {}"));
    //~^ ERROR invalid asm template string: unmatched `}` found

    // Should not ICE (test case for #131292)
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "{}/day{:02}.txt"));
    //~^ ERROR invalid asm template string: expected `}`, found `0`


    // Macro call template: should point to
    // everything within `asm!()` as error span
    asm!(concat!("abc", "r} {}"));
    //~^ ERROR invalid asm template string: unmatched `}` found


    // Literal template: should point precisely to
    // just the `}` as error span
    asm!("abc", "r} {}");
    //~^ ERROR invalid asm template string: unmatched `}` found
}

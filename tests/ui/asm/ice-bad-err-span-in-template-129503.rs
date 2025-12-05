// Regression test for ICE #129503


// Tests that we come up with decent error spans
// when the template fed to `asm!()` is itself a
// macro call like `concat!()` and should not ICE

use std::arch::asm;

fn main() {
    // Should not ICE
    asm!(concat!(r#"lJ𐏿Æ�.𐏿�"#, "r} {}"));
    //~^ ERROR invalid asm template string: unmatched `}` found


    // Macro call template: should point to
    // everything within `asm!()` as error span
    asm!(concat!("abc", "r} {}"));
    //~^ ERROR invalid asm template string: unmatched `}` found


    // Literal template: should point precisely to
    // just the `}` as error span
    asm!("abc", "r} {}");
    //~^ ERROR invalid asm template string: unmatched `}` found
}

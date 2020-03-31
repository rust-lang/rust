// check-pass
#![warn(unused_braces, unused_parens)]

fn main() {
    let _ = (7);
    //~^WARN unnecessary parentheses

    let _ = { 7 };
    //~^ WARN unnecessary braces

    if let 7 = { 7 } {
        //~^ WARN unnecessary braces
    }

    let _: [u8; { 3 }];
    //~^ WARN unnecessary braces

    // do not emit error for multiline blocks.
    let _ = {
        7
    };

    // do not emit error for unsafe blocks.
    let _ = unsafe { 7 };

    // do not emit error, as the `{` would then
    // be parsed as part of the `return`.
    if { return } {

    }
}

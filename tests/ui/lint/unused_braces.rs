//@ check-pass
//@ run-rustfix

#![warn(unused_braces, unused_parens)]
#![allow(unreachable_code, unused_unsafe)] // for rustfix

fn consume<T>(_: T) {}

fn main() {
    let _ = (7);
    //~^WARN unnecessary parentheses

    // Do not emit a lint in these cases,
    // as we have to be careful with
    // `ref` patterns.
    {
        let _ = { 7 };

        if let 7 = { 7 } { }

        match { 7 } {
            _ => (),
        }
    }

    if { true } {
        //~^ WARN unnecessary braces
    }

    while { false } {
        //~^ WARN unnecessary braces
    }

    let _: [u8; { 3 }];
    //~^ WARN unnecessary braces

    consume({ 7 });
    //~^ WARN unnecessary braces

    // Do not emit lint for multiline blocks.
    let _ = {
        7
    };

    // Do not emit lint for unsafe blocks.
    let _ = unsafe { 7 };

    // Do not emit lint, as the `{` would then
    // be parsed as part of the `return`.
    if { return } {

    }

    // regression test for https://github.com/rust-lang/rust/issues/106899
    return { println!("!") };
    //~^ WARN unnecessary braces
}

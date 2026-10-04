// Test that a `#[const_continue]` that breaks on a polymorphic constant produces an error.
// A polymorphic constant does not have a concrete value at MIR building time, and therefore the
// `#[loop_match]~ desugaring can't handle such values.
#![allow(incomplete_features)]
#![feature(loop_match)]
#![crate_type = "lib"]

trait Foo {
    const TARGET: u8;

    fn test_u8(mut state: u8) -> &'static str {
        #[loop_match]
        loop {
            state = 'blk: {
                match state {
                    0 => {
                        #[const_continue]
                        break 'blk Self::TARGET;
                        //~^ ERROR could not determine the target branch for this `#[const_continue]`
                    }

                    1 => return "bar",
                    2 => return "baz",
                    _ => unreachable!(),
                }
            }
        }
    }
}

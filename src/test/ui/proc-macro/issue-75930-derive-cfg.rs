// check-pass
// compile-flags: -Z span-debug --error-format human
// aux-build:test-macros.rs

// Regression test for issue #75930
// Tests that we cfg-strip all targets before invoking
// a derive macro
// We need '--error-format human' to stop compiletest from
// trying to interpret proc-macro output as JSON messages
// (a pretty-printed struct may cause a line to start with '{' )
// FIXME: We currently lose spans here (see issue #43081)

#[macro_use]
extern crate test_macros;

#[print_helper(a)]
#[cfg_attr(not(FALSE), allow(dead_code))]
#[print_attr]
#[derive(Print)]
#[print_helper(b)]
struct Foo<#[cfg(FALSE)] A, B> {
    #[cfg(FALSE)] first: String,
    #[cfg_attr(FALSE, deny(warnings))] second: bool,
    third: [u8; {
        #[cfg(FALSE)] struct Bar;
        #[cfg(not(FALSE))] struct Inner;
        #[cfg(FALSE)] let a = 25;
        match true {
            #[cfg(FALSE)] true => {},
            #[cfg_attr(not(FALSE), allow(warnings))] false => {},
            _ => {}
        };

        #[print_helper(should_be_removed)]
        fn removed_fn() {
            #![cfg(FALSE)]
        }

        #[print_helper(c)] #[cfg(not(FALSE))] fn kept_fn() {
            #![cfg(not(FALSE))]
            let my_val = true;
        }

        enum TupleEnum {
            Foo(
                #[cfg(FALSE)] u8,
                #[cfg(FALSE)] bool,
                #[cfg(not(FALSE))] i32,
                #[cfg(FALSE)] String, u8
            )
        }

        struct TupleStruct(
            #[cfg(FALSE)] String,
            #[cfg(not(FALSE))] i32,
            #[cfg(FALSE)] bool,
            u8
        );

        0
    }],
    #[print_helper(d)]
    fourth: B
}

fn main() {}

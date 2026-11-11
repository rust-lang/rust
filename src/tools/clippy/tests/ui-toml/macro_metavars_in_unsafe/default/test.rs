//! Tests macro_metavars_in_unsafe with default configuration
#![feature(decl_macro)]
#![warn(clippy::macro_metavars_in_unsafe)]
#![allow(clippy::no_effect, clippy::not_unsafe_ptr_arg_deref)]

#[macro_export]
macro_rules! allow_works {
    ($v:expr) => {
        #[expect(clippy::macro_metavars_in_unsafe)]
        unsafe {
            $v;
        };
    };
}

#[macro_export]
macro_rules! simple {
    ($v:expr) => {
        unsafe {
            //~^ macro_metavars_in_unsafe
            dbg!($v);
        }
    };
}

#[macro_export]
#[rustfmt::skip] // for some reason rustfmt rewrites $r#unsafe to r#u$nsafe, bug?
macro_rules! raw_symbol {
    ($r#mod:expr, $r#unsafe:expr) => {
        unsafe {
        //~^ macro_metavars_in_unsafe
            $r#mod;
        }
        $r#unsafe;
    };
}

#[macro_export]
macro_rules! multilevel_unsafe {
    ($v:expr) => {
        unsafe {
            unsafe {
                //~^ macro_metavars_in_unsafe
                $v;
            }
        }
    };
}

#[macro_export]
macro_rules! in_function {
    ($v:expr) => {
        unsafe {
            fn f() {
                // function introduces a new body, so don't lint.
                $v;
            }
        }
    };
}

#[macro_export]
macro_rules! in_function_with_unsafe {
    ($v:expr) => {
        unsafe {
            fn f() {
                unsafe {
                    //~^ macro_metavars_in_unsafe
                    $v;
                }
            }
        }
    };
}

#[macro_export]
macro_rules! const_static {
    ($c:expr, $s:expr) => {
        unsafe {
            // const and static introduces new body, don't lint
            const _X: i32 = $c;
            static _Y: i32 = $s;
        }
    };
}

#[macro_export]
macro_rules! const_generic_in_struct {
    ($inside_unsafe:expr, $outside_unsafe:expr) => {
        unsafe {
            struct Ty<
                const L: i32 = 1,
                const M: i32 = {
                    1;
                    unsafe { $inside_unsafe }
                    //~^ macro_metavars_in_unsafe
                },
                const N: i32 = { $outside_unsafe },
            >;
        }
    };
}

#[macro_export]
macro_rules! fn_with_const_generic {
    ($inside_unsafe:expr, $outside_unsafe:expr) => {
        unsafe {
            fn f<const N: usize>() {
                $outside_unsafe;
                unsafe {
                    //~^ macro_metavars_in_unsafe
                    $inside_unsafe;
                }
            }
        }
    };
}

#[macro_export]
macro_rules! variables {
    ($inside_unsafe:expr, $outside_unsafe:expr) => {
        unsafe {
            //~^ macro_metavars_in_unsafe
            $inside_unsafe;
            let inside_unsafe = 1;
            inside_unsafe;
        }
        $outside_unsafe;
        let outside_unsafe = 1;
        outside_unsafe;
    };
}

#[macro_export]
macro_rules! multiple_matchers {
    ($inside_unsafe:expr, $outside_unsafe:expr) => {
        unsafe {
        //~^ macro_metavars_in_unsafe
            $inside_unsafe;
        }
        $outside_unsafe;
    };
    ($($v:expr, $x:expr),+) => {
        $(
            $v;
            unsafe {
            //~^ macro_metavars_in_unsafe
                $x;
            }
        );+
    };
}

#[macro_export]
macro_rules! multiple_unsafe_blocks {
    ($w:expr, $x:expr, $y:expr) => {
        $w;
        unsafe {
            //~^ macro_metavars_in_unsafe
            $x;
        }
        unsafe {
            //~^ macro_metavars_in_unsafe
            $x;
            $y;
        }
    };
}

pub macro macro2_0($v:expr) {
    unsafe {
        //~^ macro_metavars_in_unsafe
        $v;
    }
}

// don't lint private macros with the default configuration
macro_rules! private_mac {
    ($v:expr) => {
        unsafe {
            $v;
        }
    };
}

// don't lint exported macros that are doc(hidden) because they also aren't part of the public API
#[macro_export]
#[doc(hidden)]
macro_rules! exported_but_hidden {
    ($v:expr) => {
        unsafe {
            $v;
        }
    };
}

// don't lint if the same metavariable is expanded in an unsafe block and then outside of one:
// unsafe {} is still needed at callsite so not problematic
#[macro_export]
macro_rules! does_require_unsafe {
    ($v:expr) => {
        unsafe {
            $v;
        }
        $v;
    };
}

#[macro_export]
macro_rules! unsafe_from_root_ctxt {
    ($v:expr) => {
        // Expands to unsafe { 1 }, but the unsafe block is from the root ctxt and not this macro,
        // so no warning.
        $v;
    };
}

// invoked from another macro, should still generate a warning
#[macro_export]
macro_rules! nested_macro_helper {
    ($v:expr) => {{
        unsafe {
            //~^ macro_metavars_in_unsafe
            $v;
        }
    }};
}

#[macro_export]
macro_rules! nested_macros {
    ($v:expr, $v2:expr) => {{
        unsafe {
            //~^ macro_metavars_in_unsafe
            nested_macro_helper!($v);
            $v;
        }
    }};
}

pub mod issue13219 {
    #[macro_export]
    macro_rules! m {
        ($e:expr) => {
            // Metavariable in a block tail expression
            unsafe { $e }
            //~^ macro_metavars_in_unsafe
        };
    }
    pub fn f(p: *const i32) -> i32 {
        m!(*p)
    }
}

#[macro_export]
macro_rules! issue14488 {
    ($e:expr) => {
        #[expect(clippy::macro_metavars_in_unsafe)]
        unsafe {
            $e
        }
    };
}

fn main() {
    allow_works!(1);
    simple!(1);
    raw_symbol!(1, 1);
    multilevel_unsafe!(1);
    in_function!(1);
    in_function_with_unsafe!(1);
    const_static!(1, 1);
    const_generic_in_struct!(1, 1);
    fn_with_const_generic!(1, 1);
    variables!(1, 1);
    multiple_matchers!(1, 1);
    multiple_matchers!(1, 1, 1, 1);
    macro2_0!(1);
    private_mac!(1);
    exported_but_hidden!(1);
    does_require_unsafe!(1);
    multiple_unsafe_blocks!(1, 1, 1);
    unsafe_from_root_ctxt!(unsafe { 1 });
    nested_macros!(1, 1);

    // These two invocations lead to two expanded unsafe blocks, each with an `#[expect]` on it.
    // Only of them gets a warning, which used to result in an unfulfilled expectation for the other
    // expanded unsafe block.
    issue14488!(1);
    issue14488!(2);
}

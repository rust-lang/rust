#[attr = MacroUse {arguments: UseAll}]
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-format-args-duplicated-captures.pp

const X: i32 = 42;

fn main() {
    let _ =
        {
            super let args = (&X, &X);
            super let args =
                [format_argument::new_display(args.0),
                        format_argument::new_display(args.1)];
            unsafe { format_arguments::new(b"\xc0\x01 \xc0\x00", &args) }
        };
}

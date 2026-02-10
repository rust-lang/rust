//! Test for <https://github.com/rust-lang/rust/issues/145739>: identifiers referring to places
//! should have their captures de-duplicated by `format_args!`, but identifiers referring to values
//! should not be de-duplicated.
extern crate std;
#[prelude_import]
use ::std::prelude::rust_2015::*;
//@ pretty-compare-only
//@ pretty-mode:hir
//@ pp-exact:hir-format-args-duplicated-captures.pp

const X: i32 = 42;

struct Struct;

static STATIC: i32 = 0;

fn main() {
    // Consts and constructors refer to values. Whether we construct them multiple times or just
    // once can be observed in some cases, so we don't de-duplicate them.
    let _ =
        {
            super let args = (&X,);
            super let args = [format_argument::new_display(args.0)];
            unsafe {
                format_arguments::new(b"\xc0\x01 \xc8\x00\x00\x00", &args)
            }
        };
    let _ =
        {
            super let args = (&Struct,);
            super let args = [format_argument::new_display(args.0)];
            unsafe {
                format_arguments::new(b"\xc0\x01 \xc8\x00\x00\x00", &args)
            }
        };

    // Variables and statics refer to places. We can de-duplicate without an observable difference.
    let x = 3;
    let _ =
        {
            super let args = (&STATIC,);
            super let args = [format_argument::new_display(args.0)];
            unsafe {
                format_arguments::new(b"\xc0\x01 \xc8\x00\x00\x00", &args)
            }
        };
    let _ =
        {
            super let args = (&x,);
            super let args = [format_argument::new_display(args.0)];
            unsafe {
                format_arguments::new(b"\xc0\x01 \xc8\x00\x00\x00", &args)
            }
        };

    // We don't de-duplicate widths or precisions since de-duplication can be observed.
    let _ =
        {
            super let args = (&x,);
            super let args =
                [format_argument::new_display(args.0),
                        format_argument::from_usize(args.0)];
            unsafe {
                format_arguments::new(b"\xd3 \x00\x00h\x01\x00\x01 \xdb \x00\x00h\x01\x00\x00\x00\x00",
                    &args)
            }
        };
    let _ =
        {
            super let args = (&0.0, &x);
            super let args =
                [format_argument::new_display(args.0),
                        format_argument::from_usize(args.1)];
            unsafe {
                format_arguments::new(b"\xe5 \x00\x00p\x01\x00\x01 \xed \x00\x00p\x01\x00\x00\x00\x00",
                    &args)
            }
        };
}

//@ check-pass

#![allow(non_upper_case_globals)]
#![feature(format_args_nl)]

static arg0: () = ();

fn main() {
    static arg1: () = ();
    format_args!("{} {:?}", 0, 1);
    format_args_nl!("{} {:?}", 0, 1);
}

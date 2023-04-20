// check-pass

#![allow(non_upper_case_globals)]

static arg0: () = ();

fn main() {
    static arg1: () = ();
    format_args!("{} {:?}", 0, 1);
}

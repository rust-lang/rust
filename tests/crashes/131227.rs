//@ known-bug: #131227
//@ compile-flags: -Zmir-opt-level=3

static mut G: () = ();

fn myfunc() -> i32 {
    let var = &raw mut G;
    if var.is_null() {
        return 0;
    }
    0
}

fn main() {
    myfunc();
}

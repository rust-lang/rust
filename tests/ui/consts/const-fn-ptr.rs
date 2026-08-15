const fn make_fn_ptr() -> fn() {
    || {}
}

static STAT: () = make_fn_ptr()();
//~^ ERROR function pointer

const CONST: () = make_fn_ptr()();
//~^ ERROR function pointer

const fn call_ptr() {
    make_fn_ptr()();
    //~^ ERROR function pointer
}

fn main() {}

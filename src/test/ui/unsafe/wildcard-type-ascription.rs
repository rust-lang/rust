// check-pass

fn foo(ptr: *const bool) {
    let _: bool = *ptr;
}

fn main() {}

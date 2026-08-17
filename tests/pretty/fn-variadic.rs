// Check that `fn foo(x: i32, ...)` does not print as `fn foo(x: i32, ..., ...)`.
// See issue #58853.

//@ pp-exact

extern "C" {
    pub fn foo(x: i32, ...);
}

pub unsafe extern "C" fn bar(_: i32, mut ap: ...) -> usize {
    ap.next_arg::<usize>()
}

fn main() {}

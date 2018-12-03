fn foo(x: &mut i32) -> i32 {
  *x = 5;
  unknown_code(&*x);
  *x // must return 5
}

fn main() {
    println!("{}", foo(&mut 0));
}

// If we replace the `*const` by `&`, my current dev version of miri
// *does* find the problem, but not for a good reason: It finds it because
// of barriers, and we shouldn't rely on unknown code using barriers.
fn unknown_code(x: *const i32) {
    unsafe { *(x as *mut i32) = 7; } //~ ERROR barrier
}

// This is not exactly right, yet.

// Ideally we should be suggesting `as_mut` for the first case,
// and suggesting to change `as_ref` to `as_mut` in the second.

fn x(cb: &mut Option<&mut dyn FnMut()>) {
    cb.map(|cb| cb());
    //~^ ERROR cannot move out of `*cb` which is behind a mutable reference
}

fn x2(cb: &mut Option<&mut dyn FnMut()>) {
    cb.as_ref().map(|cb| cb());
    //~^ ERROR cannot borrow `*cb` as mutable, as it is behind a `&` reference
}

fn main() {}

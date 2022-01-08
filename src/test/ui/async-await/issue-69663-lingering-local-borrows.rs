// edition:2018
// build-pass

use std::ops::Index;

/// A `Send + !Sync` for demonstration purposes.
struct Banana(*mut ());
unsafe impl Send for Banana {}

impl Banana {
    /// Make a static mutable reference to Banana for convenience purposes.
    ///
    /// Any potential unsoundness here is not super relevant to the issue at hand.
    fn new() -> &'static mut Banana {
        static mut BANANA: Banana = Banana(std::ptr::null_mut());
        unsafe { &mut BANANA }
    }
}

// Peach is still Send (because `impl Send for &mut T where T: Send`)
struct Peach<'a>(&'a mut Banana);

impl<'a> std::ops::Index<usize> for Peach<'a> {
    type Output = ();
    fn index(&self, _: usize) -> &() {
        &()
    }
}

async fn baz(_: &()) {}

async fn bar() {
    let peach = Peach(Banana::new());
    let r = &*peach.index(0);
    baz(r).await;
    peach.index(0); // make sure peach is retained across yield point
}

async fn bat() {
    let peach = Peach(Banana::new());
    let r = &peach[0];
    baz(r).await;
    peach[0]; // make sure peach is retained across yield point
}

fn assert_send<T: Send>(_: T) {}

pub fn main() {
    assert_send(bar());
    assert_send(bat());
}

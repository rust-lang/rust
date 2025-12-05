pub trait NSWindow: Sized {
    fn frame(self) -> () {
        unimplemented!()
    }
    fn setFrame_display_(self, display: ()) {}
}
impl NSWindow for () {}

pub struct NSRect {}

use std::ops::Deref;
struct MainThreadSafe<T = ()>(T);
impl<T> Deref for MainThreadSafe<T> {
    type Target = T;

    fn deref(&self) -> &T {
        unimplemented!()
    }
}

fn main() {
    || {
        let ns_window = MainThreadSafe(());
        // Don't record adjustments twice for `*ns_window`
        (*ns_window).frame();
        ns_window.setFrame_display_(0);
        //~^ ERROR mismatched types
    };
}

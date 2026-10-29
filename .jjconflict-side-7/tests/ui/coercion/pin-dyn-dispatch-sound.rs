use std::marker::PhantomPinned;
use std::pin::Pin;

trait MyUnpinTrait {
    fn into_pinned_type(self: Pin<&mut Self>) -> Pin<&mut PhantomPinned>;
}
impl MyUnpinTrait for PhantomPinned {
    fn into_pinned_type(self: Pin<&mut Self>) -> Pin<&mut PhantomPinned> {
        self
    }
}
impl Unpin for dyn MyUnpinTrait {} //~ ERROR E0321

// It would be unsound for this function to compile.
fn pin_it(not_yet_pinned: &mut PhantomPinned) -> Pin<&mut PhantomPinned> {
    Pin::new(not_yet_pinned as &mut dyn MyUnpinTrait).into_pinned_type()
}

fn main() {}

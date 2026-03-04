//@ aux-build:ice_trait.rs
extern crate ice_trait;
use ice_trait::ExternalTrait;

// This test ensures that the compiler doesn't crash (ICE) when explaining
// lifetimes involving cross-crate traits. See gh-153375.

struct LocalWrapper<T>(T);

impl<T: ExternalTrait> LocalWrapper<T> {
    fn do_thing(&mut self) {
        // This exercises the note_and_explain logic on an external trait.
        let _ = self.0.build_request();
    }
}

fn main() {}

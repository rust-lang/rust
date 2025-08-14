#![crate_type = "rlib"]

// Helper for testing that we get suitable warnings when lifetime
// bound change will cause breakage.

pub fn just_ref(x: &dyn Fn()) {}

pub fn ref_obj(x: &Box<dyn Fn()>) {
    // this will change to &Box<Fn()+'static>...
}

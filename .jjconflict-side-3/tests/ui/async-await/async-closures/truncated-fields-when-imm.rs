//@ edition: 2021
//@ check-pass

pub struct Struct {
    pub path: String,
}

// In `upvar.rs`, `truncate_capture_for_optimization` means that we don't actually
// capture `&(*s.path)` here, but instead just `&(*s)`, but ONLY when the upvar is
// immutable. This means that the assumption we have in `ByMoveBody` pass is wrong.
pub fn test(s: &Struct) {
    let c = async move || { let path = &s.path; };
}

fn main() {}

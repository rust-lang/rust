// build-pass (FIXME(62277): could be check-pass?)

// Tests that automatic coercions from &mut T to *mut T
// allow borrows of T to expire immediately - essentially, that
// they work identically to 'foo as *mut T'

struct SelfReference {
    self_reference: *mut SelfReference,
}

impl SelfReference {
    fn set_self_ref(&mut self) {
        self.self_reference = self;
    }
}

fn main() {}

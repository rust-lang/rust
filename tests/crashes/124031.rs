//@ known-bug: #124031

trait Trait {
    type RefTarget;
}

impl Trait for () {}

struct Other {
    data: <() as Trait>::RefTarget,
}

fn main() {
    unsafe {
        std::mem::transmute::<Option<()>, Option<&Other>>(None);
    }
}

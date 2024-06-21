trait Trait {
    type RefTarget;
}

impl Trait for () where Missing: Trait {}
//~^ ERROR cannot find type `Missing`
//~| ERROR not all trait items implemented, missing: `RefTarget`

struct Other {
    data: <() as Trait>::RefTarget,
}

fn main() {
    unsafe {
        std::mem::transmute::<Option<()>, Option<&Other>>(None);
    }
}

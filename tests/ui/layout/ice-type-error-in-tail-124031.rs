//@ normalize-stderr: "\d+ bits" -> "$$BITS bits"

// Regression test for issue #124031
// Checks that we don't ICE when the tail
// of an ADT has a type error

trait Trait {
    type RefTarget;
}

impl Trait for () {}
//~^ ERROR not all trait items implemented, missing: `RefTarget`

struct Other {
    data: <() as Trait>::RefTarget,
}

fn main() {
    unsafe {
        std::mem::transmute::<Option<()>, Option<&Other>>(None);
        //~^ ERROR cannot transmute between types of different sizes
    }
}

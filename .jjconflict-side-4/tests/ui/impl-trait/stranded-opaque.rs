trait Trait {}

impl Trait for i32 {}

// Since `Assoc` doesn't actually exist, it's "stranded", and won't show up in
// the list of opaques that may be defined by the function. Make sure we don't
// ICE in this case.
fn produce<T>() -> impl Trait<Assoc = impl Trait> {
    //~^ ERROR associated type `Assoc` not found for `Trait`
    //~| ERROR associated type `Assoc` not found for `Trait`
    16
}

fn main () {}

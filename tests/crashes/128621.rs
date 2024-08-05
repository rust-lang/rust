//@ known-bug: rust-lang/rust#128621

trait Trait {
    type Associated;
}

impl Trait for i32 {
    type Associated = i64;
}

trait Generic<T> {}

type TraitObject = dyn Generic<<i32 as Trait>::Associated>;

struct Wrap(TraitObject);

fn cast(x: *mut TraitObject) {
    x as *mut Wrap;
}

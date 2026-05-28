//@ build-pass

trait Trait {
    type Ref<'a>;
}

impl Trait for () {
    type Ref<'a> = &'a i8;
}

struct RefRef<'a, T: Trait>(&'a <T as Trait>::Ref<'a>);

fn wrap<'a, T: Trait>(reff: &'a <T as Trait>::Ref<'a>) -> RefRef<'a, T> {
    RefRef(reff)
}

fn main() {}

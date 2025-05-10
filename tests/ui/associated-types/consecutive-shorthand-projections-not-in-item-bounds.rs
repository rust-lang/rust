// FIXME: Description

fn parametrized<T: Trait<Ty: Trait>>() {
    let _: T::Ty::Ty; //~ ERROR associated type `Ty` not found for `<T as Trait>::Ty`
}

trait Trait {
    type Ty;
}

fn main() {}

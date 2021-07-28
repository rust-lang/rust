// check-pass

trait Trait {}

impl<T0> Trait for T0 {}

impl<T1: ?Sized> Id for T1 {
    type Assoc = T1;
}

trait Id {
    type Assoc: ?Sized;
}

struct NewType<T2: ?Sized + Id>(T2::Assoc);

fn coerce_newtype_slice<'a, T3, const N: usize>(array: &'a NewType<[T3; N]>) -> &'a NewType<[T3]> {
    array
}

fn coerce_newtype_trait<T4: Trait + 'static>(tr: &NewType<T4>) -> &NewType<dyn Trait> {
    tr
}

fn main() {
    let nt = NewType::<[i32; 1]>([0]);
    coerce_newtype_slice(&nt);
    coerce_newtype_trait(&nt);
}

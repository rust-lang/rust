//@ known-bug: rust-lang/rust#146371
pub trait Trait {
    type Assoc: Copy + 'static;
}

const fn conjure<T>() -> T {
    panic!()
}

const fn get_assoc<T: Trait>() -> impl Copy {
    conjure::<<T as Trait>::Assoc>()
}

pub fn foo<T: Trait<Assoc = i32> + Trait<Assoc = i64>>() {
    const {
        get_assoc::<T>();
    }
}

pub fn main() {}

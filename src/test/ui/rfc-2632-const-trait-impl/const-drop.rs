#![feature(const_trait_impl)]
#![feature(const_fn_trait_bound)]
#![feature(const_mut_refs)]
#![feature(const_panic)]

struct S;

impl const Drop for S {
    fn drop(&mut self) {
        // NB: There is no way to tell that a const destructor is ran,
        // because even if we can operate on mutable variables, it will
        // not be reflected because everything is `const`. So we panic
        // here, attempting to make the CTFE engine error.
        panic!("much const drop")
        //~^ ERROR evaluation of constant value failed
    }
}

const fn a<T: ~const Drop>(_: T) {}

const _: () = a(S);

fn main() {}

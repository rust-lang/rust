// revisions: stock precise
#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
#![feature(const_fn_trait_bound)]
#![cfg_attr(precise, feature(const_precise_live_drops))]

struct NonTrivialDrop;

impl Drop for NonTrivialDrop {
    fn drop(&mut self) {
        println!("Non trivial drop");
    }
}

struct ConstImplWithDropGlue(NonTrivialDrop);

impl const Drop for ConstImplWithDropGlue {
    fn drop(&mut self) {}
}

const fn check<T: ~const Drop>(_: T) {}

macro_rules! check_all {
    ($($exp:expr),*$(,)?) => {$(
        const _: () = check($exp);
    )*};
}

check_all! {
    NonTrivialDrop,
    //~^ ERROR the trait bound
    ConstImplWithDropGlue(NonTrivialDrop),
    //~^ ERROR the trait bound
}

fn main() {}

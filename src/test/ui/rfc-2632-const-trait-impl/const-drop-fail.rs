#![feature(const_trait_impl)]
#![feature(const_mut_refs)]
#![feature(const_fn_trait_bound)]

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

const fn check<T: ~const Drop>() {}

macro_rules! check_all {
    ($($T:ty),*$(,)?) => {$(
        const _: () = check::<$T>();  
    )*};
}

check_all! {
    ConstImplWithDropGlue,
}

fn main() {}

// Check that we emit an error when an erroneous constant is used in a pattern, even if this
// pattern is uninhabited.

#![feature(never_type)]
#![feature(inline_const_pat)]
//~^ WARN the feature `inline_const_pat` is incomplete

fn foo<T>(x: Result<T, !>) -> T {
    match x {
        Ok(y) => y,
        Err(const { panic!() }) => panic!(),
        //~^ ERROR evaluation of `foo::<T>::{constant#0}` failed
    }
}

fn main() {}

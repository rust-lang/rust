// https://github.com/rust-lang/rust-clippy/issues/12302
use std::marker::PhantomData;

pub struct Aa<T>(PhantomData<T>);

fn aa(a: Aa<String) {
//~^ ERROR: expected one of

}

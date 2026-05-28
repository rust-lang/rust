//! Auxiliary crate testing this issue https://github.com/rust-lang/rust/issues/4208
#![crate_name="generic_fn_with_supertrait_bound_cross_crate"]
#![crate_type = "lib"]

pub trait Trig<T> {
    fn sin(&self) -> T;
}

pub fn sin<T:Trig<R>, R>(theta: &T) -> R { theta.sin() }

pub trait Angle<T>: Trig<T> {}

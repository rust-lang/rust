//@ revisions: normalize_param_env normalize_obligation hrtb
//@ check-pass
//@ compile-flags: -Znext-solver
#![feature(rustc_attrs)]
#![rustc_no_implicit_bounds]

trait Foo {
    #[cfg(normalize_param_env)]
    type Gat<'a> where <Self as Mirror>::Assoc: 'a;
    #[cfg(normalize_obligation)]
    type Gat<'a> where Self: 'a;
    #[cfg(hrtb)]
    type Gat<'b> where for<'a> <Self as MirrorRegion<'a>>::Assoc: 'b;
}

trait Mirror { type Assoc; }
impl<T> Mirror for T { type Assoc = T; }

trait MirrorRegion<'a> { type Assoc; }
impl<'a, T> MirrorRegion<'a> for T { type Assoc = T; }

impl<T> Foo for T {
    #[cfg(normalize_param_env)]
    type Gat<'a> = i32 where T: 'a;
    #[cfg(normalize_obligation)]
    type Gat<'a> = i32 where <T as Mirror>::Assoc: 'a;
    #[cfg(hrtb)]
    type Gat<'b> = i32 where Self: 'b;
}

fn main() {}

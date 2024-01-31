// revisions: normalize_param_env normalize_obligation
// check-pass
// compile-flags: -Znext-solver

trait Foo {
    #[cfg(normalize_param_env)]
    type Gat<'a> where <Self as Mirror>::Assoc: 'a;
    #[cfg(normalize_obligation)]
    type Gat<'a> where Self: 'a;
}

trait Mirror { type Assoc: ?Sized; }
impl<T: ?Sized> Mirror for T { type Assoc = T; }

impl<T> Foo for T {
    #[cfg(normalize_param_env)]
    type Gat<'a> = i32 where T: 'a;
    #[cfg(normalize_obligation)]
    type Gat<'a> = i32 where <T as Mirror>::Assoc: 'a;
}

fn main() {}

// Regression test for #55756.
//
// In this test, the result of `self.callee` is a projection `<D as
// Database<'?0>>::Guard`. As it may contain a destructor, the dropck
// rules require that this type outlivess the scope of `state`. Unfortunately,
// our region inference is not smart enough to figure out how to
// translate a requirement like
//
//     <D as Database<'0>>::guard: 'r
//
// into a requirement that `'0: 'r` -- in particular, it fails to do
// so because it *also* knows that `<D as Database<'a>>::Guard: 'a`
// from the trait definition. Faced with so many choices, the current
// solver opts to do nothing.
//
// The problem we were having was that we would observe two
// potentially applicable rules when trying to find bounds for `<T as
// Database<'0>>::Guard`:
//
// ```
// <T as Database<'a>>::Guard: 'a // from the where clauses
// for<'b> { <T as Database<'b>>::Guard: 'b } // from the trait definition
// ```
//
// Because of this, we decided to take no action to influence
// inference, which means that `'0` winds up unconstrained, leading to
// the ultimate error.
//
// compile-pass

#![crate_type="lib"]

pub trait Database<'a> {
    type Guard: 'a;
}

pub struct Stateful<'a, D: 'a>(&'a D);

impl<'b, D: for <'a> Database<'a>> Stateful<'b, D> {
    pub fn callee<'a>(&'a self) -> <D as Database<'a>>::Guard {
        unimplemented!()
    }
    pub fn caller<'a>(&'a self) -> <D as Database<'a>>::Guard {
        let state = self.callee();
        self.callee()
    }
}

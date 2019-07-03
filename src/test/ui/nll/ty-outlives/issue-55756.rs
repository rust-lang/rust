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
// Fixed by tweaking the solver to recognize that the constraint from
// the environment duplicates one from the trait.
//
// build-pass (FIXME(62277): could be check-pass?)

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
        unimplemented!()
    }
}

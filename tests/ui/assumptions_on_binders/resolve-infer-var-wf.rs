//@ compile-flags: -Znext-solver=globally -Zassumptions-on-binders
//@ check-pass

// Computing placeholder assumptions when entering a binder visits every
// type and const in the goal predicate and computes their WF obligations.
// The visited terms may contain resolvable inference variables; the solver
// must resolve them first, otherwise computing WF obligations ICEs with
// `assertion failed: term == infcx.resolve_vars_if_possible(term)`.

use std::marker::PhantomData;

fn hint_app<TArg, TRet>(f: &dyn Fn(TArg) -> TRet) -> &dyn Fn(TArg) -> TRet {
    f
}

enum List<'a, A> {
    Nil(PhantomData<&'a A>),
}

enum Tree<'a> {
    Leaf(PhantomData<&'a ()>),
}

type Priqueue<'a> = &'a List<'a, &'a Tree<'a>>;

struct Program;

impl<'a> Program {
    fn alloc<T>(&'a self, t: T) -> &'a T {
        todo!()
    }

    fn unzip(
        &'a self,
        t: &'a Tree,
        cont: &dyn Fn(Priqueue) -> Priqueue,
    ) -> Priqueue<'a> {
        match t {
            _ => hint_app(cont)(self.alloc(List::Nil(PhantomData))),
        }
    }
}

fn main() {}

//@ run-pass
#![allow(dead_code)]
#![allow(unused_variables)]
// This is an example where the older inference algorithm failed. The
// specifics of why it failed are somewhat, but not entirely, tailed
// to the algorithm. Ultimately the problem is that when computing the
// mutual supertype of both sides of the `if` it would be faced with a
// choice of tightening bounds or unifying variables and it took the
// wrong path. The new algorithm avoids this problem and hence this
// example typechecks correctly.


enum ScopeChain<'a> {
    Link(Scope<'a>),
    End
}

type Scope<'a> = &'a ScopeChain<'a>;

struct OuterContext;

struct Context<'a> {
    foo: &'a OuterContext
}

impl<'a> Context<'a> {
    fn foo(&mut self, scope: Scope) {
        let link = if 1 < 2 {
            let l = ScopeChain::Link(scope);
            self.take_scope(&l);
            l
        } else {
            ScopeChain::Link(scope)
        };
        self.take_scope(&link);
    }

    fn take_scope(&mut self, x: Scope) {
    }
}

fn main() { }

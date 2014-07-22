// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
        let link = if 1i < 2 {
            let l = Link(scope);
            self.take_scope(&l);
            l
        } else {
            Link(scope)
        };
        self.take_scope(&link);
    }

    fn take_scope(&mut self, x: Scope) {
    }
}

fn main() { }

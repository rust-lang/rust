// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S;

impl S {
    fn late<'a, 'b>(self, _: &'a u8, _: &'b u8) {}
    fn late_implicit(self, _: &u8, _: &u8) {}
    fn early<'a, 'b>(self) -> (&'a u8, &'b u8) { loop {} }
    fn late_early<'a, 'b>(self, _: &'a u8) -> &'b u8 { loop {} }
    fn late_implicit_early<'b>(self, _: &u8) -> &'b u8 { loop {} }
    fn late_implicit_self_early<'b>(&self) -> &'b u8 { loop {} }
    fn late_unused_early<'a, 'b>(self) -> &'b u8 { loop {} }
    fn life_and_type<'a, T>(self) -> &'a T { loop {} }
}

fn method_call() {
    S.early(); // OK
    S.early::<'static>();
    //~^ ERROR expected 2 lifetime parameters, found 1 lifetime parameter
    S.early::<'static, 'static, 'static>();
    //~^ ERROR expected at most 2 lifetime parameters, found 3 lifetime parameters
    let _: &u8 = S.life_and_type::<'static>();
    S.life_and_type::<u8>();
    S.life_and_type::<'static, u8>();
}

fn ufcs() {
    S::late(S, &0, &0); // OK
    S::late::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late::<'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late::<'static, 'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_early(S, &0); // OK
    S::late_early::<'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_early::<'static, 'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly

    S::late_implicit(S, &0, &0); // OK
    S::late_implicit::<'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit::<'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit::<'static, 'static, 'static>(S, &0, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_early(S, &0); // OK
    S::late_implicit_early::<'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_early::<'static, 'static, 'static>(S, &0);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_self_early(&S); // OK
    S::late_implicit_self_early::<'static, 'static>(&S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_implicit_self_early::<'static, 'static, 'static>(&S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_unused_early(S); // OK
    S::late_unused_early::<'static, 'static>(S);
    //~^ ERROR cannot specify lifetime arguments explicitly
    S::late_unused_early::<'static, 'static, 'static>(S);
    //~^ ERROR cannot specify lifetime arguments explicitly

    S::early(S); // OK
    S::early::<'static>(S);
    //~^ ERROR expected 2 lifetime parameters, found 1 lifetime parameter
    S::early::<'static, 'static, 'static>(S);
    //~^ ERROR expected at most 2 lifetime parameters, found 3 lifetime parameters
    let _: &u8 = S::life_and_type::<'static>(S);
    S::life_and_type::<u8>(S);
    S::life_and_type::<'static, u8>(S);
}

fn main() {}

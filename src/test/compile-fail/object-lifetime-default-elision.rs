// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test various cases where the old rules under lifetime elision
// yield slightly different results than the new rules.

#![allow(dead_code)]

trait SomeTrait {
    fn dummy(&self) { }
}

struct SomeStruct<'a> {
    r: Box<SomeTrait+'a>
}

fn deref<T>(ss: &T) -> T {
    // produces the type of a deref without worrying about whether a
    // move out would actually be legal
    loop { }
}

fn load0<'a>(ss: &'a Box<SomeTrait>) -> Box<SomeTrait> {
    // Under old rules, the fully elaborated types of input/output were:
    //
    // for<'a,'b> fn(&'a Box<SomeTrait+'b>) -> Box<SomeTrait+'a>
    //
    // Under new rules the result is:
    //
    // for<'a> fn(&'a Box<SomeTrait+'static>) -> Box<SomeTrait+'static>
    //
    // Therefore, no type error.

    deref(ss)
}

fn load1(ss: &SomeTrait) -> &SomeTrait {
    // Under old rules, the fully elaborated types of input/output were:
    //
    // for<'a,'b> fn(&'a (SomeTrait+'b)) -> &'a (SomeTrait+'a)
    //
    // Under new rules the result is:
    //
    // for<'a> fn(&'a (SomeTrait+'a)) -> &'a (SomeTrait+'a)
    //
    // In both cases, returning `ss` is legal.

    ss
}

fn load2<'a>(ss: &'a SomeTrait) -> &SomeTrait {
    // Same as `load1` but with an explicit name thrown in for fun.

    ss
}

fn load3<'a,'b>(ss: &'a SomeTrait) -> &'b SomeTrait {
    // Under old rules, the fully elaborated types of input/output were:
    //
    // for<'a,'b,'c>fn(&'a (SomeTrait+'c)) -> &'b (SomeTrait+'a)
    //
    // Based on the input/output types, the compiler could infer that
    //     'c : 'a
    //     'b : 'a
    // must hold, and therefore it permitted `&'a (Sometrait+'c)` to be
    // coerced to `&'b (SomeTrait+'a)`.
    //
    // Under the newer defaults, though, we get:
    //
    // for<'a,'b> fn(&'a (SomeTrait+'a)) -> &'b (SomeTrait+'b)
    //
    // which fails to type check.

    ss
        //~^ ERROR cannot infer
        //~| ERROR cannot infer
}

fn main() {
}

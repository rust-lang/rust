// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/// The result of a generator resumption.
#[derive(Debug)]
#[cfg(not(stage0))]
#[lang = "generator_state"]
#[unstable(feature = "generator_trait", issue = "0")]
pub enum State<Y, R> {
    /// The generator suspended with a value.
    Yielded(Y),

    /// The generator completed with a return value.
    Complete(R),
}

/// The trait implemented by builtin generator types.
#[cfg(not(stage0))]
#[lang = "generator"]
#[unstable(feature = "generator_trait", issue = "0")]
#[fundamental]
pub trait Generator<Arg = ()> {
    /// The type of value this generator yields.
    type Yield;

    /// The type of value this generator returns.
    type Return;

    /// This resumes the execution of the generator.
    fn resume(&mut self, arg: Arg) -> State<Self::Yield, Self::Return>;
}

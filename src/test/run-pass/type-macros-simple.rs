// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! Tuple {
    { $A:ty,$B:ty } => { ($A, $B) }
}

fn main() {
    let x: Tuple!(i32, i32) = (1, 2);
}

fn issue_36540() {
    let i32 = 0;
    macro_rules! m { () => { i32 } }
    struct S<T = m!()>(m!(), T) where T: Trait<m!()>;

    let x: m!() = m!();
    std::cell::Cell::<m!()>::new(m!());
    impl<T> std::ops::Index<m!()> for Trait<(m!(), T)>
        where T: Trait<m!()>
    {
        type Output = m!();
        fn index(&self, i: m!()) -> &m!() {
            unimplemented!()
        }
    }
}

trait Trait<T> {}

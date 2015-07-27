// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn invariant_id<'a,'b>(t: &'b mut &'static ()) -> &'b mut &'a ()
    where 'a: 'static { t }
fn static_id<'a>(t: &'a ()) -> &'static ()
    where 'a: 'static { t }
fn static_id_indirect<'a,'b>(t: &'a ()) -> &'static ()
    where 'a: 'b, 'b: 'static { t }
fn ref_id<'a>(t: &'a ()) -> &'a () where 'static: 'a { t }

static UNIT: () = ();

fn main()
{
    let mut val : &'static () = &UNIT;
    invariant_id(&mut val);
    static_id(val);
    static_id_indirect(val);
    ref_id(val);
}

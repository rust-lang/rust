// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[deny(visible_local_types)];
#[allow(dead_code, unused_variable)];

pub use Reexport = Reexported;

struct Private;
struct Reexported;
pub struct Public;


trait PrivTrait {
    fn ok_1(&self, ok_1: Public, ok_2: Reexported, ok_3: Private) {}
    fn ok_2(&self, ok_1: Public, ok_2: Reexported, ok_3: Private);
}

impl PrivTrait for Public {
    fn ok_2(&self, ok_1: Public, ok_2: Reexported, ok_3: Private) {}
}

pub trait PubTrait {
    fn err_1(&self,
             ok_1: Public,
             ok_2: Reexported,
             err: Private) {} //~ ERROR non-exported type used in exported method signature

    fn err_2(&self, err: Private);  //~ ERROR non-exported type used in exported method signature
}

impl PubTrait for Option<Private> {
    fn err_2(&self, ok: Private) {}
}

impl Private {
    pub fn ok() -> Private { Private }
}
impl Public {
    pub fn err() -> Private { //~ ERROR non-exported type used in exported method signature
        Private
    }
}

pub struct PublicStruct {
    priv ok_1: Private,

    ok_2: Public,
    ok_3: Reexported,
    err: Private //~ ERROR non-exported type used in exported struct field
}

struct PrivateStruct {
    ok_1: Public,
    ok_2: Private
}


pub enum PublicEnum {
    priv PubOk1(Private),

    PubOk2(Public),
    PubOk3(Reexported),
    PubErr(Private) //~ ERROR non-exported type used in exported enum variant
}

enum PrivateEnum {
    PrivOk1(Private),
    PrivOk2(Public),

    pub PrivOk3(Public),
    pub PrivOk4(Reexported),
    pub PrivErr(Private) // FIXME #11680 (ERROR non-exported type used in exported enum variant)
}

pub fn public_fn<Ok: PrivTrait>(
    ok_1: Public,
    ok_2: &PrivTrait,
    ok_3: Reexported,
    err_1: Private, //~ ERROR non-exported type used in exported function signature
    err_2: || -> Private //~ ERROR non-exported type used in exported function signature
        ) -> Option<Private> { //~ ERROR non-exported type used in exported function signature
    None
}

fn private_fn<Ok: PrivTrait>(
    ok_1: Public,
    ok_2: Private,
    ok_3: &PrivTrait,
    ok_4: || -> Private
        ) -> Option<Private> {
    None
}

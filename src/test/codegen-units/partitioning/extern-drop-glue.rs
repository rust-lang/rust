// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-tidy-linelength
// compile-flags:-Zprint-trans-items=lazy

#![allow(dead_code)]
#![crate_type="lib"]

// aux-build:cgu_extern_drop_glue.rs
extern crate cgu_extern_drop_glue;

//~ TRANS_ITEM drop-glue cgu_extern_drop_glue::Struct[0] @@ extern_drop_glue[OnceODR] extern_drop_glue-mod1[OnceODR]

struct LocalStruct(cgu_extern_drop_glue::Struct);

//~ TRANS_ITEM fn extern_drop_glue::user[0] @@ extern_drop_glue[WeakODR]
fn user()
{
    //~ TRANS_ITEM drop-glue extern_drop_glue::LocalStruct[0] @@ extern_drop_glue[OnceODR]
    let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
}

mod mod1 {
    use cgu_extern_drop_glue;

    struct LocalStruct(cgu_extern_drop_glue::Struct);

    //~ TRANS_ITEM fn extern_drop_glue::mod1[0]::user[0] @@ extern_drop_glue-mod1[WeakODR]
    fn user()
    {
        //~ TRANS_ITEM drop-glue extern_drop_glue::mod1[0]::LocalStruct[0] @@ extern_drop_glue-mod1[OnceODR]
        let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
    }
}


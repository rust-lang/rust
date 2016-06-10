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

// We specify -Z incremental here because we want to test the partitioning for
// incremental compilation
// compile-flags:-Zprint-trans-items=lazy -Zincremental=tmp/partitioning-tests/extern-drop-glue

#![allow(dead_code)]
#![crate_type="lib"]

// aux-build:cgu_extern_drop_glue.rs
extern crate cgu_extern_drop_glue;

//~ TRANS_ITEM drop-glue cgu_extern_drop_glue::Struct[0] @@ extern_drop_glue[Internal] extern_drop_glue-mod1[Internal]
//~ TRANS_ITEM drop-glue-contents cgu_extern_drop_glue::Struct[0] @@ extern_drop_glue[Internal] extern_drop_glue-mod1[Internal]

struct LocalStruct(cgu_extern_drop_glue::Struct);

//~ TRANS_ITEM fn extern_drop_glue::user[0] @@ extern_drop_glue[External]
fn user()
{
    //~ TRANS_ITEM drop-glue extern_drop_glue::LocalStruct[0] @@ extern_drop_glue[Internal]
    let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
}

mod mod1 {
    use cgu_extern_drop_glue;

    struct LocalStruct(cgu_extern_drop_glue::Struct);

    //~ TRANS_ITEM fn extern_drop_glue::mod1[0]::user[0] @@ extern_drop_glue-mod1[External]
    fn user()
    {
        //~ TRANS_ITEM drop-glue extern_drop_glue::mod1[0]::LocalStruct[0] @@ extern_drop_glue-mod1[Internal]
        let _ = LocalStruct(cgu_extern_drop_glue::Struct(0));
    }
}

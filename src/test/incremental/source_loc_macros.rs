// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// This test makes sure that different expansions of the file!(), line!(),
// column!() macros get picked up by the incr. comp. hash.

// revisions:rpass1 rpass2

// compile-flags: -Z query-dep-graph

#![feature(rustc_attrs)]

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn line_same() {
    let _ = line!();
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn col_same() {
    let _ = column!();
}

#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_clean(label="HirBody", cfg="rpass2")]
fn file_same() {
    let _ = file!();
}

#[cfg(rpass1)]
fn line_different() {
    let _ = line!();
}

#[cfg(rpass2)]
#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_dirty(label="HirBody", cfg="rpass2")]
fn line_different() {
    let _ = line!();
}

#[cfg(rpass1)]
fn col_different() {
    let _ = column!();
}

#[cfg(rpass2)]
#[rustc_clean(label="Hir", cfg="rpass2")]
#[rustc_dirty(label="HirBody", cfg="rpass2")]
fn col_different() {
    let _ =        column!();
}

fn main() {
    line_same();
    line_different();
    col_same();
    col_different();
    file_same();
}

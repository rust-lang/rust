// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// This test case tests the incremental compilation hash (ICH) implementation
// for struct constructor expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// must-compile-successfully
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


struct RegularStruct {
    x: i32,
    y: i64,
    z: i16,
}

// Change field value (regular struct) -----------------------------------------
#[cfg(cfail1)]
fn change_field_value_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 0,
        y: 1,
        z: 2,
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_field_value_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 0,
        y: 2,
        z: 2,
    }
}



// Change field order (regular struct) -----------------------------------------
#[cfg(cfail1)]
fn change_field_order_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_field_order_regular_struct() -> RegularStruct {
    RegularStruct {
        y: 4,
        x: 3,
        z: 5,
    }
}



// Add field (regular struct) --------------------------------------------------
#[cfg(cfail1)]
fn add_field_regular_struct() -> RegularStruct {
    let struct1 = RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    };

    RegularStruct {
        x: 7,
        .. struct1
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn add_field_regular_struct() -> RegularStruct {
    let struct1 = RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    };

    RegularStruct {
        x: 7,
        y: 8,
        .. struct1
    }
}



// Change field label (regular struct) -----------------------------------------
#[cfg(cfail1)]
fn change_field_label_regular_struct() -> RegularStruct {
    let struct1 = RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    };

    RegularStruct {
        x: 7,
        y: 9,
        .. struct1
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_field_label_regular_struct() -> RegularStruct {
    let struct1 = RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    };

    RegularStruct {
        x: 7,
        z: 9,
        .. struct1
    }
}



struct RegularStruct2 {
    x: i8,
    y: i8,
    z: i8,
}

// Change constructor path (regular struct) ------------------------------------
#[cfg(cfail1)]
fn change_constructor_path_regular_struct() {
    let _ = RegularStruct {
        x: 0,
        y: 1,
        z: 2,
    };
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_constructor_path_regular_struct() {
    let _ = RegularStruct2 {
        x: 0,
        y: 1,
        z: 2,
    };
}



// Change constructor path indirectly (regular struct) -------------------------
mod change_constructor_path_indirectly_regular_struct {
    #[cfg(cfail1)]
    use super::RegularStruct as Struct;
    #[cfg(not(cfail1))]
    use super::RegularStruct2 as Struct;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn function() -> Struct {
        Struct {
            x: 0,
            y: 1,
            z: 2,
        }
    }
}



struct TupleStruct(i32, i64, i16);

// Change field value (tuple struct) -------------------------------------------
#[cfg(cfail1)]
fn change_field_value_tuple_struct() -> TupleStruct {
    TupleStruct(0, 1, 2)
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_field_value_tuple_struct() -> TupleStruct {
    TupleStruct(0, 1, 3)
}



struct TupleStruct2(u16, u16, u16);

// Change constructor path (tuple struct) --------------------------------------
#[cfg(cfail1)]
fn change_constructor_path_tuple_struct() {
    let _ = TupleStruct(0, 1, 2);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_metadata_clean(cfg="cfail2")]
#[rustc_metadata_clean(cfg="cfail3")]
fn change_constructor_path_tuple_struct() {
    let _ = TupleStruct2(0, 1, 2);
}



// Change constructor path indirectly (tuple struct) ---------------------------
mod change_constructor_path_indirectly_tuple_struct {
    #[cfg(cfail1)]
    use super::TupleStruct as Struct;
    #[cfg(not(cfail1))]
    use super::TupleStruct2 as Struct;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_metadata_dirty(cfg="cfail2")]
    #[rustc_metadata_clean(cfg="cfail3")]
    fn function() -> Struct {
        Struct(0, 1, 2)
    }
}

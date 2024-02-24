// This test case tests the incremental compilation hash (ICH) implementation
// for struct constructor expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [cfail1]compile-flags: -Zincremental-ignore-spans
//@ [cfail2]compile-flags: -Zincremental-ignore-spans
//@ [cfail3]compile-flags: -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


pub struct RegularStruct {
    x: i32,
    y: i64,
    z: i16,
}

// Change field value (regular struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_field_value_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 0,
        y: 1,
        z: 2,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_field_value_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 0,
        y: 2,
        z: 2,
    }
}



// Change field order (regular struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_field_order_regular_struct() -> RegularStruct {
    RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,typeck,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_field_order_regular_struct() -> RegularStruct {
    RegularStruct {
        y: 4,
        x: 3,
        z: 5,
    }
}



// Add field (regular struct)
#[cfg(any(cfail1,cfail4))]
pub fn add_field_regular_struct() -> RegularStruct {
    let struct1 = RegularStruct {
        x: 3,
        y: 4,
        z: 5,
    };

    RegularStruct {
        x: 7,
        // --
        .. struct1
    }
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn add_field_regular_struct() -> RegularStruct {
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



// Change field label (regular struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_field_label_regular_struct() -> RegularStruct {
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

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_field_label_regular_struct() -> RegularStruct {
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



pub struct RegularStruct2 {
    x: i8,
    y: i8,
    z: i8,
}

// Change constructor path (regular struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_constructor_path_regular_struct() {
    let _ = RegularStruct  {
        x: 0,
        y: 1,
        z: 2,
    };
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_constructor_path_regular_struct() {
    let _ = RegularStruct2 {
        x: 0,
        y: 1,
        z: 2,
    };
}



// Change constructor path indirectly (regular struct)
pub mod change_constructor_path_indirectly_regular_struct {
    #[cfg(any(cfail1,cfail4))]
    use super::RegularStruct as Struct;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::RegularStruct2 as Struct;

    #[rustc_clean(cfg="cfail2", except="fn_sig,opt_hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="fn_sig,opt_hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail6")]
    pub fn function() -> Struct {
        Struct {
            x: 0,
            y: 1,
            z: 2,
        }
    }
}



pub struct TupleStruct(i32, i64, i16);

// Change field value (tuple struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_field_value_tuple_struct() -> TupleStruct {
    TupleStruct(0, 1, 2)
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,optimized_mir")]
#[rustc_clean(cfg="cfail6")]
pub fn change_field_value_tuple_struct() -> TupleStruct {
    TupleStruct(0, 1, 3)
}



pub struct TupleStruct2(u16, u16, u16);

// Change constructor path (tuple struct)
#[cfg(any(cfail1,cfail4))]
pub fn change_constructor_path_tuple_struct() {
    let _ = TupleStruct (0, 1, 2);
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,typeck")]
#[rustc_clean(cfg="cfail6")]
pub fn change_constructor_path_tuple_struct() {
    let _ = TupleStruct2(0, 1, 2);
}



// Change constructor path indirectly (tuple struct)
pub mod change_constructor_path_indirectly_tuple_struct {
    #[cfg(any(cfail1,cfail4))]
    use super::TupleStruct as Struct;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::TupleStruct2 as Struct;

    #[rustc_clean(cfg="cfail5", except="fn_sig,opt_hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail6")]
    #[rustc_clean(cfg="cfail2", except="fn_sig,opt_hir_owner_nodes,optimized_mir,typeck")]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> Struct {
        Struct(0, 1, 2)
    }
}

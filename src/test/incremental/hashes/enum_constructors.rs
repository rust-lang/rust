// This test case tests the incremental compilation hash (ICH) implementation
// for struct constructor expressions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


pub enum Enum {
    Struct {
        x: i32,
        y: i64,
        z: i16,
    },
    Tuple(i32, i64, i16)
}

// Change field value (struct-like) -----------------------------------------
#[cfg(cfail1)]
pub fn change_field_value_struct_like() -> Enum {
    Enum::Struct {
        x: 0,
        y: 1,
        z: 2,
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
#[rustc_clean(cfg="cfail3")]
pub fn change_field_value_struct_like() -> Enum {
    Enum::Struct {
        x: 0,
        y: 2,
        z: 2,
    }
}



// Change field order (struct-like) -----------------------------------------
#[cfg(cfail1)]
pub fn change_field_order_struct_like() -> Enum {
    Enum::Struct {
        x: 3,
        y: 4,
        z: 5,
    }
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,typeck_tables_of")]
#[rustc_clean(cfg="cfail3")]
// FIXME(michaelwoerister):Interesting. I would have thought that that changes the MIR. And it
// would if it were not all constants
pub fn change_field_order_struct_like() -> Enum {
    Enum::Struct {
        y: 4,
        x: 3,
        z: 5,
    }
}


pub enum Enum2 {
    Struct {
        x: i8,
        y: i8,
        z: i8,
    },
    Struct2 {
        x: i8,
        y: i8,
        z: i8,
    },
    Tuple(u16, u16, u16),
    Tuple2(u64, u64, u64),
}

// Change constructor path (struct-like) ------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_path_struct_like() {
    let _ = Enum::Struct {
        x: 0,
        y: 1,
        z: 2,
    };
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built,typeck_tables_of")]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_path_struct_like() {
    let _ = Enum2::Struct {
        x: 0,
        y: 1,
        z: 2,
    };
}



// Change variant (regular struct) ------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_variant_struct_like() {
    let _ = Enum2::Struct {
        x: 0,
        y: 1,
        z: 2,
    };
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_variant_struct_like() {
    let _ = Enum2::Struct2 {
        x: 0,
        y: 1,
        z: 2,
    };
}


// Change constructor path indirectly (struct-like) -------------------------
pub mod change_constructor_path_indirectly_struct_like {
    #[cfg(cfail1)]
    use super::Enum as TheEnum;
    #[cfg(not(cfail1))]
    use super::Enum2 as TheEnum;

    #[rustc_clean(
        cfg="cfail2",
        except="fn_sig,Hir,HirBody,optimized_mir,mir_built,\
                typeck_tables_of"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> TheEnum {
        TheEnum::Struct {
            x: 0,
            y: 1,
            z: 2,
        }
    }
}


// Change constructor variant indirectly (struct-like) ---------------------------
pub mod change_constructor_variant_indirectly_struct_like {
    use super::Enum2;
    #[cfg(cfail1)]
    use super::Enum2::Struct as Variant;
    #[cfg(not(cfail1))]
    use super::Enum2::Struct2 as Variant;

    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> Enum2 {
        Variant {
            x: 0,
            y: 1,
            z: 2,
        }
    }
}


// Change field value (tuple-like) -------------------------------------------
#[cfg(cfail1)]
pub fn change_field_value_tuple_like() -> Enum {
    Enum::Tuple(0, 1, 2)
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
#[rustc_clean(cfg="cfail3")]
pub fn change_field_value_tuple_like() -> Enum {
    Enum::Tuple(0, 1, 3)
}



// Change constructor path (tuple-like) --------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_path_tuple_like() {
    let _ = Enum::Tuple(0, 1, 2);
}

#[cfg(not(cfail1))]
#[rustc_clean(
    cfg="cfail2",
    except="HirBody,optimized_mir,mir_built,typeck_tables_of"
)]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_path_tuple_like() {
    let _ = Enum2::Tuple(0, 1, 2);
}



// Change constructor variant (tuple-like) --------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_variant_tuple_like() {
    let _ = Enum2::Tuple(0, 1, 2);
}

#[cfg(not(cfail1))]
#[rustc_clean(
    cfg="cfail2",
    except="HirBody,optimized_mir,mir_built,typeck_tables_of"
)]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_variant_tuple_like() {
    let _ = Enum2::Tuple2(0, 1, 2);
}


// Change constructor path indirectly (tuple-like) ---------------------------
pub mod change_constructor_path_indirectly_tuple_like {
    #[cfg(cfail1)]
    use super::Enum as TheEnum;
    #[cfg(not(cfail1))]
    use super::Enum2 as TheEnum;

    #[rustc_clean(
        cfg="cfail2",
        except="fn_sig,Hir,HirBody,optimized_mir,mir_built,\
                typeck_tables_of"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> TheEnum {
        TheEnum::Tuple(0, 1, 2)
    }
}



// Change constructor variant indirectly (tuple-like) ---------------------------
pub mod change_constructor_variant_indirectly_tuple_like {
    use super::Enum2;
    #[cfg(cfail1)]
    use super::Enum2::Tuple as Variant;
    #[cfg(not(cfail1))]
    use super::Enum2::Tuple2 as Variant;

    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built,typeck_tables_of")]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> Enum2 {
        Variant(0, 1, 2)
    }
}


pub enum Clike {
    A,
    B,
    C
}

pub enum Clike2 {
    B,
    C,
    D
}

// Change constructor path (C-like) --------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_path_c_like() {
    let _ = Clike::B;
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built,typeck_tables_of")]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_path_c_like() {
    let _ = Clike2::B;
}



// Change constructor variant (C-like) --------------------------------------
#[cfg(cfail1)]
pub fn change_constructor_variant_c_like() {
    let _ = Clike::A;
}

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
#[rustc_clean(cfg="cfail3")]
pub fn change_constructor_variant_c_like() {
    let _ = Clike::C;
}


// Change constructor path indirectly (C-like) ---------------------------
pub mod change_constructor_path_indirectly_c_like {
    #[cfg(cfail1)]
    use super::Clike as TheEnum;
    #[cfg(not(cfail1))]
    use super::Clike2 as TheEnum;

    #[rustc_clean(
        cfg="cfail2",
        except="fn_sig,Hir,HirBody,optimized_mir,mir_built,\
                typeck_tables_of"
    )]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> TheEnum {
        TheEnum::B
    }
}



// Change constructor variant indirectly (C-like) ---------------------------
pub mod change_constructor_variant_indirectly_c_like {
    use super::Clike;
    #[cfg(cfail1)]
    use super::Clike::A as Variant;
    #[cfg(not(cfail1))]
    use super::Clike::B as Variant;

    #[rustc_clean(cfg="cfail2", except="HirBody,optimized_mir,mir_built")]
    #[rustc_clean(cfg="cfail3")]
    pub fn function() -> Clike {
        Variant
    }
}

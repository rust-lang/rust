// This test case tests the incremental compilation hash (ICH) implementation
// for `type` definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for `type` definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the enum's metadata, and that it stays
// the same between rev2 and rev3.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]


// Change type (primitive) -----------------------------------------------------
#[cfg(cfail1)]
type ChangePrimitiveType = i32;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangePrimitiveType = i64;



// Change mutability -----------------------------------------------------------
#[cfg(cfail1)]
type ChangeMutability = &'static i32;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangeMutability = &'static mut i32;



// Change mutability -----------------------------------------------------------
#[cfg(cfail1)]
type ChangeLifetime<'a> = (&'static i32, &'a i32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type ChangeLifetime<'a> = (&'a i32, &'a i32);



// Change type (struct) -----------------------------------------------------------
struct Struct1;
struct Struct2;

#[cfg(cfail1)]
type ChangeTypeStruct = Struct1;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangeTypeStruct = Struct2;



// Change type (tuple) ---------------------------------------------------------
#[cfg(cfail1)]
type ChangeTypeTuple = (u32, u64);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangeTypeTuple = (u32, i64);



// Change type (enum) ----------------------------------------------------------
enum Enum1 {
    Var1,
    Var2,
}
enum Enum2 {
    Var1,
    Var2,
}

#[cfg(cfail1)]
type ChangeTypeEnum = Enum1;

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangeTypeEnum = Enum2;



// Add tuple field -------------------------------------------------------------
#[cfg(cfail1)]
type AddTupleField = (i32, i64);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type AddTupleField = (i32, i64, i16);



// Change nested tuple field ---------------------------------------------------
#[cfg(cfail1)]
type ChangeNestedTupleField = (i32, (i64, i16));

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner")]
#[rustc_clean(cfg="cfail3")]
type ChangeNestedTupleField = (i32, (i64, i8));



// Add type param --------------------------------------------------------------
#[cfg(cfail1)]
type AddTypeParam<T1> = (T1, T1);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddTypeParam<T1, T2> = (T1, T2);



// Add type param bound --------------------------------------------------------
#[cfg(cfail1)]
type AddTypeParamBound<T1> = (T1, u32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddTypeParamBound<T1: Clone> = (T1, u32);



// Add type param bound in where clause ----------------------------------------
#[cfg(cfail1)]
type AddTypeParamBoundWhereClause<T1> where T1: Clone = (T1, u32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddTypeParamBoundWhereClause<T1> where T1: Clone+Copy = (T1, u32);



// Add lifetime param ----------------------------------------------------------
#[cfg(cfail1)]
type AddLifetimeParam<'a> = (&'a u32, &'a u32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddLifetimeParam<'a, 'b> = (&'a u32, &'b u32);



// Add lifetime param bound ----------------------------------------------------
#[cfg(cfail1)]
type AddLifetimeParamBound<'a, 'b> = (&'a u32, &'b u32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddLifetimeParamBound<'a, 'b: 'a> = (&'a u32, &'b u32);



// Add lifetime param bound in where clause ------------------------------------
#[cfg(cfail1)]
type AddLifetimeParamBoundWhereClause<'a, 'b, 'c>
where 'b: 'a
    = (&'a u32, &'b u32, &'c u32);

#[cfg(not(cfail1))]
#[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
type AddLifetimeParamBoundWhereClause<'a, 'b, 'c>
where 'b: 'a,
      'c: 'a
    = (&'a u32, &'b u32, &'c u32);



// Change Trait Bound Indirectly -----------------------------------------------
trait ReferencedTrait1 {}
trait ReferencedTrait2 {}

mod change_trait_bound_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
    #[rustc_clean(cfg="cfail3")]
    type ChangeTraitBoundIndirectly<T: Trait> = (T, u32);
}



// Change Trait Bound Indirectly In Where Clause -------------------------------
mod change_trait_bound_indirectly_in_where_clause {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg="cfail2", except="hir_owner,hir_owner_nodes")]
    #[rustc_clean(cfg="cfail3")]
    type ChangeTraitBoundIndirectly<T> where T : Trait = (T, u32);
}

// This test case tests the incremental compilation hash (ICH) implementation
// for struct definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for struct definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the struct's metadata, and that it stays
// the same between rev2 and rev3.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
// compile-flags: -Z query-dep-graph
// [cfail1]compile-flags: -Zincremental-ignore-spans
// [cfail2]compile-flags: -Zincremental-ignore-spans
// [cfail3]compile-flags: -Zincremental-ignore-spans
// [cfail4]compile-flags: -Zincremental-relative-spans
// [cfail5]compile-flags: -Zincremental-relative-spans
// [cfail6]compile-flags: -Zincremental-relative-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Layout ----------------------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
pub struct LayoutPacked;

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
#[repr(packed)]
pub struct LayoutPacked;

#[cfg(any(cfail1,cfail4))]
struct LayoutC;

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
#[repr(C)]
struct LayoutC;


// Tuple Struct Change Field Type ----------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct TupleStructFieldType(i32);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
// Note that changing the type of a field does not change the type of the struct or enum, but
// adding/removing fields or changing a fields name or visibility does.
struct TupleStructFieldType(
    u32
);


// Tuple Struct Add Field ------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct TupleStructAddField(i32);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct TupleStructAddField(
    i32,
    u32
);


// Tuple Struct Field Visibility -----------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct TupleStructFieldVisibility(char);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct TupleStructFieldVisibility(pub char);


// Record Struct Field Type ----------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct RecordStructFieldType { x: f32 }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
// Note that changing the type of a field does not change the type of the struct or enum, but
// adding/removing fields or changing a fields name or visibility does.
struct RecordStructFieldType {
    x: u64
}


// Record Struct Field Name ----------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct RecordStructFieldName { x: f32 }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct RecordStructFieldName { y: f32 }


// Record Struct Add Field -----------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct RecordStructAddField { x: f32 }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct RecordStructAddField {
    x: f32,
    y: () }


// Record Struct Field Visibility ----------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct RecordStructFieldVisibility { x: f32 }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct RecordStructFieldVisibility {
    pub x: f32
}


// Add Lifetime Parameter ------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct AddLifetimeParameter<'a>(&'a f32, &'a f64);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of,generics_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of,generics_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddLifetimeParameter<'a, 'b>(&'a f32, &'b f64);


// Add Lifetime Parameter Bound ------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct AddLifetimeParameterBound<'a, 'b>(&'a f32, &'b f64);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddLifetimeParameterBound<'a, 'b: 'a>(
    &'a f32,
    &'b f64
);

#[cfg(any(cfail1,cfail4))]
struct AddLifetimeParameterBoundWhereClause<'a, 'b>(&'a f32, &'b f64);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddLifetimeParameterBoundWhereClause<'a, 'b>(
    &'a f32,
    &'b f64)
    where 'b: 'a;


// Add Type Parameter ----------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct AddTypeParameter<T1>(T1, T1);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of,generics_of,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,type_of,generics_of,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddTypeParameter<T1, T2>(
     // The field contains the parent's Generics, so it's dirty even though its
     // type hasn't changed.
    T1,
    T2
);


// Add Type Parameter Bound ----------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct AddTypeParameterBound<T>(T);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddTypeParameterBound<T: Send>(
    T
);


#[cfg(any(cfail1,cfail4))]
struct AddTypeParameterBoundWhereClause<T>(T);

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
struct AddTypeParameterBoundWhereClause<T>(
    T
) where T: Sync;


// Empty struct ----------------------------------------------------------------
// Since we cannot change anything in this case, we just make sure that the
// fingerprint is stable (i.e., that there are no random influences like memory
// addresses taken into account by the hashing algorithm).
// Note: there is no #[cfg(...)], so this is ALWAYS compiled
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub struct EmptyStruct;


// Visibility ------------------------------------------------------------------

#[cfg(any(cfail1,cfail4))]
struct Visibility;

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
pub struct Visibility;

struct ReferencedType1;
struct ReferencedType2;

// Tuple Struct Change Field Type Indirectly -----------------------------------
mod tuple_struct_change_field_type_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedType2 as FieldType;

    #[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    struct TupleStruct(
        FieldType
    );
}


// Record Struct Change Field Type Indirectly -----------------------------------
mod record_struct_change_field_type_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedType2 as FieldType;

    #[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    struct RecordStruct {
        _x: FieldType
    }
}




trait ReferencedTrait1 {}
trait ReferencedTrait2 {}

// Change Trait Bound Indirectly -----------------------------------------------
mod change_trait_bound_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    struct Struct<T: Trait>(T);
}

// Change Trait Bound Indirectly In Where Clause -------------------------------
mod change_trait_bound_indirectly_in_where_clause {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail2")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(except="hir_owner,hir_owner_nodes,predicates_of", cfg="cfail5")]
    #[rustc_clean(cfg="cfail6")]
    struct Struct<T>(T) where T : Trait;
}

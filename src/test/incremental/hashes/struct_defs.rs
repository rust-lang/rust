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
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans


#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]

// Layout ----------------------------------------------------------------------
#[cfg(cfail1)]
pub struct LayoutPacked;

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
#[repr(packed)]
pub struct LayoutPacked;

#[cfg(cfail1)]
struct LayoutC;

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
#[repr(C)]
struct LayoutC;


// Tuple Struct Change Field Type ----------------------------------------------

#[cfg(cfail1)]
struct TupleStructFieldType(i32);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
// Note that changing the type of a field does not change the type of the struct or enum, but
// adding/removing fields or changing a fields name or visibility does.
struct TupleStructFieldType(
    u32
);


// Tuple Struct Add Field ------------------------------------------------------

#[cfg(cfail1)]
struct TupleStructAddField(i32);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct TupleStructAddField(
    i32,
    u32
);


// Tuple Struct Field Visibility -----------------------------------------------

#[cfg(cfail1)]
struct TupleStructFieldVisibility(char);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct TupleStructFieldVisibility(pub char);


// Record Struct Field Type ----------------------------------------------------

#[cfg(cfail1)]
struct RecordStructFieldType { x: f32 }

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
// Note that changing the type of a field does not change the type of the struct or enum, but
// adding/removing fields or changing a fields name or visibility does.
struct RecordStructFieldType {
    x: u64
}


// Record Struct Field Name ----------------------------------------------------

#[cfg(cfail1)]
struct RecordStructFieldName { x: f32 }

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct RecordStructFieldName { y: f32 }


// Record Struct Add Field -----------------------------------------------------

#[cfg(cfail1)]
struct RecordStructAddField { x: f32 }

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct RecordStructAddField {
    x: f32,
    y: () }


// Record Struct Field Visibility ----------------------------------------------

#[cfg(cfail1)]
struct RecordStructFieldVisibility { x: f32 }

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct RecordStructFieldVisibility {
    pub x: f32
}


// Add Lifetime Parameter ------------------------------------------------------

#[cfg(cfail1)]
struct AddLifetimeParameter<'a>(&'a f32, &'a f64);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_dirty(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddLifetimeParameter<'a, 'b>(&'a f32, &'b f64);


// Add Lifetime Parameter Bound ------------------------------------------------

#[cfg(cfail1)]
struct AddLifetimeParameterBound<'a, 'b>(&'a f32, &'b f64);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_dirty(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddLifetimeParameterBound<'a, 'b: 'a>(
    &'a f32,
    &'b f64
);

#[cfg(cfail1)]
struct AddLifetimeParameterBoundWhereClause<'a, 'b>(&'a f32, &'b f64);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_dirty(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddLifetimeParameterBoundWhereClause<'a, 'b>(
    &'a f32,
    &'b f64)
    where 'b: 'a;


// Add Type Parameter ----------------------------------------------------------

#[cfg(cfail1)]
struct AddTypeParameter<T1>(T1, T1);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_dirty(label="type_of", cfg="cfail2")]
#[rustc_dirty(label="generics_of", cfg="cfail2")]
#[rustc_dirty(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddTypeParameter<T1, T2>(
     // The field contains the parent's Generics, so it's dirty even though its
     // type hasn't changed.
    T1,
    T2
);


// Add Type Parameter Bound ----------------------------------------------------

#[cfg(cfail1)]
struct AddTypeParameterBound<T>(T);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_dirty(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddTypeParameterBound<T: Send>(
    T
);


#[cfg(cfail1)]
struct AddTypeParameterBoundWhereClause<T>(T);

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_dirty(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
struct AddTypeParameterBoundWhereClause<T>(
    T
) where T: Sync;


// Empty struct ----------------------------------------------------------------
// Since we cannot change anything in this case, we just make sure that the
// fingerprint is stable (i.e., that there are no random influences like memory
// addresses taken into account by the hashing algorithm).
// Note: there is no #[cfg(...)], so this is ALWAYS compiled
#[rustc_clean(label="Hir", cfg="cfail2")]
#[rustc_clean(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
pub struct EmptyStruct;


// Visibility ------------------------------------------------------------------

#[cfg(cfail1)]
struct Visibility;

#[cfg(not(cfail1))]
#[rustc_dirty(label="Hir", cfg="cfail2")]
#[rustc_dirty(label="HirBody", cfg="cfail2")]
#[rustc_clean(label="type_of", cfg="cfail2")]
#[rustc_clean(label="generics_of", cfg="cfail2")]
#[rustc_clean(label="predicates_of", cfg="cfail2")]
#[rustc_clean(label="Hir", cfg="cfail3")]
#[rustc_clean(label="HirBody", cfg="cfail3")]
#[rustc_clean(label="type_of", cfg="cfail3")]
#[rustc_clean(label="generics_of", cfg="cfail3")]
#[rustc_clean(label="predicates_of", cfg="cfail3")]
pub struct Visibility;

struct ReferencedType1;
struct ReferencedType2;

// Tuple Struct Change Field Type Indirectly -----------------------------------
mod tuple_struct_change_field_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as FieldType;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="type_of", cfg="cfail2")]
    #[rustc_clean(label="generics_of", cfg="cfail2")]
    #[rustc_clean(label="predicates_of", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_clean(label="type_of", cfg="cfail3")]
    #[rustc_clean(label="generics_of", cfg="cfail3")]
    #[rustc_clean(label="predicates_of", cfg="cfail3")]
    struct TupleStruct(
        FieldType
    );
}


// Record Struct Change Field Type Indirectly -----------------------------------
mod record_struct_change_field_type_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(cfail1))]
    use super::ReferencedType2 as FieldType;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="type_of", cfg="cfail2")]
    #[rustc_clean(label="generics_of", cfg="cfail2")]
    #[rustc_clean(label="predicates_of", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_clean(label="type_of", cfg="cfail3")]
    #[rustc_clean(label="generics_of", cfg="cfail3")]
    #[rustc_clean(label="predicates_of", cfg="cfail3")]
    struct RecordStruct {
        _x: FieldType
    }
}




trait ReferencedTrait1 {}
trait ReferencedTrait2 {}

// Change Trait Bound Indirectly -----------------------------------------------
mod change_trait_bound_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="type_of", cfg="cfail2")]
    #[rustc_clean(label="generics_of", cfg="cfail2")]
    #[rustc_dirty(label="predicates_of", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_clean(label="type_of", cfg="cfail3")]
    #[rustc_clean(label="generics_of", cfg="cfail3")]
    #[rustc_clean(label="predicates_of", cfg="cfail3")]
    struct Struct<T: Trait>(T);
}

// Change Trait Bound Indirectly In Where Clause -------------------------------
mod change_trait_bound_indirectly_in_where_clause {
    #[cfg(cfail1)]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_dirty(label="Hir", cfg="cfail2")]
    #[rustc_dirty(label="HirBody", cfg="cfail2")]
    #[rustc_clean(label="type_of", cfg="cfail2")]
    #[rustc_clean(label="generics_of", cfg="cfail2")]
    #[rustc_dirty(label="predicates_of", cfg="cfail2")]
    #[rustc_clean(label="Hir", cfg="cfail3")]
    #[rustc_clean(label="HirBody", cfg="cfail3")]
    #[rustc_clean(label="type_of", cfg="cfail3")]
    #[rustc_clean(label="generics_of", cfg="cfail3")]
    #[rustc_clean(label="predicates_of", cfg="cfail3")]
    struct Struct<T>(T) where T : Trait;
}

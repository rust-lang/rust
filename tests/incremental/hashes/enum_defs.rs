// This test case tests the incremental compilation hash (ICH) implementation
// for enum definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for enum definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the enum's metadata, and that it stays
// the same between rev2 and rev3.

//@ build-pass (FIXME(62277): could be check-pass?)
//@ revisions: cfail1 cfail2 cfail3 cfail4 cfail5 cfail6
//@ compile-flags: -Z query-dep-graph -O
//@ [cfail1]compile-flags: -Zincremental-ignore-spans
//@ [cfail2]compile-flags: -Zincremental-ignore-spans
//@ [cfail3]compile-flags: -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![feature(stmt_expr_attributes)]
#![crate_type="rlib"]



// Change enum visibility -----------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum     EnumVisibility { A }

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
pub enum EnumVisibility { A }



// Change name of a c-style variant -------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeNameCStyleVariant {
    Variant1,
    Variant2,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeNameCStyleVariant {
    Variant1,
    Variant2Changed,
}



// Change name of a tuple-style variant ---------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeNameTupleStyleVariant {
    Variant1,
    Variant2(u32, f32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeNameTupleStyleVariant {
    Variant1,
    Variant2Changed(u32, f32),
}



// Change name of a struct-style variant --------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeNameStructStyleVariant {
    Variant1,
    Variant2 { a: u32, b: f32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeNameStructStyleVariant {
    Variant1,
    Variant2Changed { a: u32, b: f32 },
}



// Change the value of a c-style variant --------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeValueCStyleVariant0 {
    Variant1,
    Variant2 = 11,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeValueCStyleVariant0 {
    Variant1,
    Variant2 = 22,
}

#[cfg(any(cfail1,cfail4))]
enum EnumChangeValueCStyleVariant1 {
    Variant1,
    Variant2,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeValueCStyleVariant1 {
    Variant1,
    Variant2 = 11,
}



// Add a c-style variant ------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddCStyleVariant {
    Variant1,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddCStyleVariant {
    Variant1,
    Variant2,
}



// Remove a c-style variant ---------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumRemoveCStyleVariant {
    Variant1,
    Variant2,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumRemoveCStyleVariant {
    Variant1,
}



// Add a tuple-style variant --------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddTupleStyleVariant {
    Variant1,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddTupleStyleVariant {
    Variant1,
    Variant2(u32, f32),
}



// Remove a tuple-style variant -----------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumRemoveTupleStyleVariant {
    Variant1,
    Variant2(u32, f32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumRemoveTupleStyleVariant {
    Variant1,
}



// Add a struct-style variant -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddStructStyleVariant {
    Variant1,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddStructStyleVariant {
    Variant1,
    Variant2 { a: u32, b: f32 },
}



// Remove a struct-style variant ----------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumRemoveStructStyleVariant {
    Variant1,
    Variant2 { a: u32, b: f32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumRemoveStructStyleVariant {
    Variant1,
}



// Change the type of a field in a tuple-style variant ------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeFieldTypeTupleStyleVariant {
    Variant1(u32, u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeFieldTypeTupleStyleVariant {
    Variant1(u32,
        u64),
}



// Change the type of a field in a struct-style variant -----------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeFieldTypeStructStyleVariant {
    Variant1,
    Variant2 { a: u32, b: u32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeFieldTypeStructStyleVariant {
    Variant1,
    Variant2 {
        a: u32,
        b: u64
    },
}



// Change the name of a field in a struct-style variant -----------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeFieldNameStructStyleVariant {
    Variant1 { a: u32, b: u32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeFieldNameStructStyleVariant {
    Variant1 { a: u32, c: u32 },
}



// Change order of fields in a tuple-style variant ----------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeOrderTupleStyleVariant {
    Variant1(u32, u64),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeOrderTupleStyleVariant {
    Variant1(
        u64,
        u32),
}



// Change order of fields in a struct-style variant ---------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeFieldOrderStructStyleVariant {
    Variant1 { a: u32, b: f32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeFieldOrderStructStyleVariant {
    Variant1 { b: f32, a: u32 },
}



// Add a field to a tuple-style variant ---------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddFieldTupleStyleVariant {
    Variant1(u32, u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddFieldTupleStyleVariant {
    Variant1(u32, u32, u32),
}



// Add a field to a struct-style variant --------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddFieldStructStyleVariant {
    Variant1 { a: u32, b: u32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddFieldStructStyleVariant {
    Variant1 { a: u32, b: u32, c: u32 },
}



// Add #[must_use] to the enum ------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddMustUse {
    Variant1,
    Variant2,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5")]
#[rustc_clean(cfg="cfail6")]
#[must_use]
enum EnumAddMustUse {
    Variant1,
    Variant2,
}



// Add #[repr(C)] to the enum -------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddReprC {
    Variant1,
    Variant2,
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="type_of")]
#[rustc_clean(cfg="cfail6")]
#[repr(C)]
enum EnumAddReprC {
    Variant1,
    Variant2,
}



// Change the name of a type parameter ----------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeNameOfTypeParameter<S> {
    Variant1(S),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeNameOfTypeParameter<T> {
    Variant1(T),
}



// Add a type parameter ------------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddTypeParameter<S> {
    Variant1(S),
    Variant2(S),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddTypeParameter<S, T> {
    Variant1(S),
    Variant2(T),
}



// Change the name of a lifetime parameter ------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumChangeNameOfLifetimeParameter<'a> {
    Variant1(&'a u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumChangeNameOfLifetimeParameter<'b> {
    Variant1(&'b u32),
}



// Add a lifetime parameter ---------------------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddLifetimeParameter<'a> {
    Variant1(&'a u32),
    Variant2(&'a u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddLifetimeParameter<'a, 'b> {
    Variant1(&'a u32),
    Variant2(&'b u32),
}



// Add a lifetime bound to a lifetime parameter -------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddLifetimeParameterBound<'a, 'b> {
    Variant1(&'a u32),
    Variant2(&'b u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddLifetimeParameterBound<'a, 'b: 'a> {
    Variant1(&'a u32),
    Variant2(&'b u32),
}

// Add a lifetime bound to a type parameter -----------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddLifetimeBoundToParameter<'a, T> {
    Variant1(T),
    Variant2(&'a u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddLifetimeBoundToParameter<'a, T: 'a> {
    Variant1(T),
    Variant2(&'a u32),
}



// Add a trait bound to a type parameter --------------------------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddTraitBound<S> {
    Variant1(S),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddTraitBound<T: Sync> {
    Variant1(T),
}



// Add a lifetime bound to a lifetime parameter in where clause ---------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddLifetimeParameterBoundWhere<'a, 'b> {
    Variant1(&'a u32),
    Variant2(&'b u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddLifetimeParameterBoundWhere<'a, 'b> where 'b: 'a {
    Variant1(&'a u32),
    Variant2(&'b u32),
}



// Add a lifetime bound to a type parameter in where clause -------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddLifetimeBoundToParameterWhere<'a, T> {
    Variant1(T),
    Variant2(&'a u32),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddLifetimeBoundToParameterWhere<'a, T> where T: 'a {
    Variant1(T),
    Variant2(&'a u32),
}



// Add a trait bound to a type parameter in where clause ----------------------
#[cfg(any(cfail1,cfail4))]
enum EnumAddTraitBoundWhere<S> {
    Variant1(S),
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,generics_of,predicates_of,type_of")]
#[rustc_clean(cfg="cfail6")]
enum EnumAddTraitBoundWhere<T> where T: Sync {
    Variant1(T),
}



// In an enum with two variants, swap usage of type parameters ----------------
#[cfg(any(cfail1,cfail4))]
enum EnumSwapUsageTypeParameters<A, B> {
    Variant1 { a: A },
    Variant2 { a: B },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumSwapUsageTypeParameters<A, B> {
    Variant1 {
        a: B
    },
    Variant2 {
        a: A
    },
}



// In an enum with two variants, swap usage of lifetime parameters ------------
#[cfg(any(cfail1,cfail4))]
enum EnumSwapUsageLifetimeParameters<'a, 'b> {
    Variant1 { a: &'a u32 },
    Variant2 { b: &'b u32 },
}

#[cfg(not(any(cfail1,cfail4)))]
#[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail3")]
#[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
#[rustc_clean(cfg="cfail6")]
enum EnumSwapUsageLifetimeParameters<'a, 'b> {
    Variant1 {
        a: &'b u32
    },
    Variant2 {
        b: &'a u32
    },
}



struct ReferencedType1;
struct ReferencedType2;



// Change field type in tuple-style variant indirectly by modifying a use statement
mod change_field_type_indirectly_tuple_style {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedType2 as FieldType;

    #[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
    #[rustc_clean(cfg="cfail6")]
    enum TupleStyle {
        Variant1(
            FieldType
        )
    }
}



// Change field type in record-style variant indirectly by modifying a use statement
mod change_field_type_indirectly_struct_style {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedType1 as FieldType;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedType2 as FieldType;

    #[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes")]
    #[rustc_clean(cfg="cfail6")]
    enum StructStyle {
        Variant1 {
            a: FieldType
        }
    }
}



trait ReferencedTrait1 {}
trait ReferencedTrait2 {}



// Change trait bound of type parameter indirectly by modifying a use statement
mod change_trait_bound_indirectly {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
    #[rustc_clean(cfg="cfail6")]
    enum Enum<T: Trait> {
        Variant1(T)
    }
}



// Change trait bound of type parameter in where clause indirectly by modifying a use statement
mod change_trait_bound_indirectly_where {
    #[cfg(any(cfail1,cfail4))]
    use super::ReferencedTrait1 as Trait;
    #[cfg(not(any(cfail1,cfail4)))]
    use super::ReferencedTrait2 as Trait;

    #[rustc_clean(cfg="cfail2", except="opt_hir_owner_nodes,predicates_of")]
    #[rustc_clean(cfg="cfail3")]
    #[rustc_clean(cfg="cfail5", except="opt_hir_owner_nodes,predicates_of")]
    #[rustc_clean(cfg="cfail6")]
    enum Enum<T> where T: Trait {
        Variant1(T)
    }
}

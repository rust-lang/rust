// This test case tests the incremental compilation hash (ICH) implementation
// for trait definitions.

// The general pattern followed here is: Change one thing between rev1 and rev2
// and make sure that the hash has changed, then change nothing between rev2 and
// rev3 and make sure that the hash has not changed.

// We also test the ICH for trait definitions exported in metadata. Same as
// above, we want to make sure that the change between rev1 and rev2 also
// results in a change of the ICH for the trait's metadata, and that it stays
// the same between rev2 and rev3.

// build-pass (FIXME(62277): could be check-pass?)
// revisions: cfail1 cfail2 cfail3
// compile-flags: -Z query-dep-graph -Zincremental-ignore-spans

#![allow(warnings)]
#![feature(rustc_attrs)]
#![crate_type="rlib"]
#![feature(associated_type_defaults)]


// Change trait visibility
#[cfg(cfail1)]
trait TraitVisibility { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
pub trait TraitVisibility { }



// Change trait unsafety
#[cfg(cfail1)]
trait TraitUnsafety { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
unsafe trait TraitUnsafety { }



// Add method
#[cfg(cfail1)]
trait TraitAddMethod {
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
pub trait TraitAddMethod {
    fn method();
}



// Change name of method
#[cfg(cfail1)]
trait TraitChangeMethodName {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeMethodName {
    fn methodChanged();
}



// Add return type to method
#[cfg(cfail1)]
trait TraitAddReturnType {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddReturnType {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method() -> u32;
}



// Change return type of method
#[cfg(cfail1)]
trait TraitChangeReturnType {
    fn method() -> u32;
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeReturnType {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method() -> u64;
}



// Add parameter to method
#[cfg(cfail1)]
trait TraitAddParameterToMethod {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddParameterToMethod {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(a: u32);
}



// Change name of method parameter
#[cfg(cfail1)]
trait TraitChangeMethodParameterName {
    fn method(a: u32);
    fn with_default(x: i32) {}
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeMethodParameterName {
    // FIXME(#38501) This should preferably always be clean.
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(b: u32);

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    #[rustc_dirty(label="hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(label="hir_owner_nodes", cfg="cfail3")]
    fn with_default(y: i32) {}
}



// Change type of method parameter (i32 => i64)
#[cfg(cfail1)]
trait TraitChangeMethodParameterType {
    fn method(a: i32);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeMethodParameterType {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(a: i64);
}



// Change type of method parameter (&i32 => &mut i32)
#[cfg(cfail1)]
trait TraitChangeMethodParameterTypeRef {
    fn method(a: &i32);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeMethodParameterTypeRef {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(a: &mut i32);
}



// Change order of method parameters
#[cfg(cfail1)]
trait TraitChangeMethodParametersOrder {
    fn method(a: i32, b: i64);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeMethodParametersOrder {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(b: i64, a: i32);
}



// Add default implementation to method
#[cfg(cfail1)]
trait TraitAddMethodAutoImplementation {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddMethodAutoImplementation {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method() { }
}



// Change order of methods
#[cfg(cfail1)]
trait TraitChangeOrderOfMethods {
    fn method0();
    fn method1();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeOrderOfMethods {
    fn method1();
    fn method0();
}



// Change mode of self parameter
#[cfg(cfail1)]
trait TraitChangeModeSelfRefToMut {
    fn method(&self);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeModeSelfRefToMut {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(&mut self);
}



#[cfg(cfail1)]
trait TraitChangeModeSelfOwnToMut: Sized {
    fn method(self) {}
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeModeSelfOwnToMut: Sized {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    #[rustc_dirty(label="hir_owner_nodes", cfg="cfail2")]
    #[rustc_clean(label="hir_owner_nodes", cfg="cfail3")]
    fn method(mut self) {}
}



#[cfg(cfail1)]
trait TraitChangeModeSelfOwnToRef {
    fn method(self);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeModeSelfOwnToRef {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method(&self);
}



// Add unsafe modifier to method
#[cfg(cfail1)]
trait TraitAddUnsafeModifier {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddUnsafeModifier {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    unsafe fn method();
}



// Add extern modifier to method
#[cfg(cfail1)]
trait TraitAddExternModifier {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddExternModifier {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    extern "C" fn method();
}



// Change extern "C" to extern "stdcall"
#[cfg(cfail1)]
trait TraitChangeExternCToRustIntrinsic {
    extern "C" fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeExternCToRustIntrinsic {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    extern "stdcall" fn method();
}



// Add type parameter to method
#[cfg(cfail1)]
trait TraitAddTypeParameterToMethod {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTypeParameterToMethod {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<T>();
}



// Add lifetime parameter to method
#[cfg(cfail1)]
trait TraitAddLifetimeParameterToMethod {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeParameterToMethod {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<'a>();
}



// dummy trait for bound
trait ReferencedTrait0 { }
trait ReferencedTrait1 { }

// Add trait bound to method type parameter
#[cfg(cfail1)]
trait TraitAddTraitBoundToMethodTypeParameter {
    fn method<T>();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitBoundToMethodTypeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<T: ReferencedTrait0>();
}



// Add builtin bound to method type parameter
#[cfg(cfail1)]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    fn method<T>();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltinBoundToMethodTypeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<T: Sized>();
}



// Add lifetime bound to method lifetime parameter
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    fn method<'a, 'b>(a: &'a u32, b: &'b u32);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToMethodLifetimeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<'a, 'b: 'a>(a: &'a u32, b: &'b u32);
}



// Add second trait bound to method type parameter
#[cfg(cfail1)]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    fn method<T: ReferencedTrait0>();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondTraitBoundToMethodTypeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<T: ReferencedTrait0 + ReferencedTrait1>();
}



// Add second builtin bound to method type parameter
#[cfg(cfail1)]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    fn method<T: Sized>();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondBuiltinBoundToMethodTypeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<T: Sized + Sync>();
}



// Add second lifetime bound to method lifetime parameter
#[cfg(cfail1)]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    fn method<'a, 'b, 'c: 'a>(a: &'a u32, b: &'b u32, c: &'c u32);
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeBoundToMethodLifetimeParameter {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method<'a, 'b, 'c: 'a + 'b>(a: &'a u32, b: &'b u32, c: &'c u32);
}



// Add associated type
#[cfg(cfail1)]
trait TraitAddAssociatedType {

    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddAssociatedType {
    type Associated;

    fn method();
}



// Add trait bound to associated type
#[cfg(cfail1)]
trait TraitAddTraitBoundToAssociatedType {
    type Associated;

    fn method();
}


// Apparently the type bound contributes to the predicates of the trait, but
// does not change the associated item itself.
#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitBoundToAssociatedType {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    type Associated: ReferencedTrait0;

    fn method();
}



// Add lifetime bound to associated type
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    type Associated;

    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToAssociatedType<'a> {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    type Associated: 'a;

    fn method();
}



// Add default to associated type
#[cfg(cfail1)]
trait TraitAddDefaultToAssociatedType {
    type Associated;

    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddDefaultToAssociatedType {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    type Associated = ReferenceType0;

    fn method();
}



// Add associated constant
#[cfg(cfail1)]
trait TraitAddAssociatedConstant {
    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddAssociatedConstant {
    const Value: u32;

    fn method();
}



// Add initializer to associated constant
#[cfg(cfail1)]
trait TraitAddInitializerToAssociatedConstant {
    const Value: u32;

    fn method();
}

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddInitializerToAssociatedConstant {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    const Value: u32 = 1;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method();
}



// Change type of associated constant
#[cfg(cfail1)]
trait TraitChangeTypeOfAssociatedConstant {
    const Value: u32;

    fn method();
}

#[cfg(not(cfail1))]
#[rustc_clean(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitChangeTypeOfAssociatedConstant {
    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    const Value: f64;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    fn method();
}



// Add super trait
#[cfg(cfail1)]
trait TraitAddSuperTrait { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSuperTrait : ReferencedTrait0 { }



// Add builtin bound (Send or Copy)
#[cfg(cfail1)]
trait TraitAddBuiltiBound { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltiBound : Send { }



// Add 'static lifetime bound to trait
#[cfg(cfail1)]
trait TraitAddStaticLifetimeBound { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddStaticLifetimeBound : 'static { }



// Add super trait as second bound
#[cfg(cfail1)]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitAsSecondBound : ReferencedTrait0 + ReferencedTrait1 { }

#[cfg(cfail1)]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitAsSecondBoundFromBuiltin : Send + ReferencedTrait0 { }



// Add builtin bound as second bound
#[cfg(cfail1)]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltinBoundAsSecondBound : ReferencedTrait0 + Send { }

#[cfg(cfail1)]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltinBoundAsSecondBoundFromBuiltin: Send + Copy { }



// Add 'static bounds as second bound
#[cfg(cfail1)]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddStaticBoundAsSecondBound : ReferencedTrait0 + 'static { }

#[cfg(cfail1)]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddStaticBoundAsSecondBoundFromBuiltin : Send + 'static { }



// Add type parameter to trait
#[cfg(cfail1)]
trait TraitAddTypeParameterToTrait { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTypeParameterToTrait<T> { }



// Add lifetime parameter to trait
#[cfg(cfail1)]
trait TraitAddLifetimeParameterToTrait { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeParameterToTrait<'a> { }



// Add trait bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddTraitBoundToTypeParameterOfTrait<T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }



// Add lifetime bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToTypeParameterOfTrait<'a, T: 'a> { }



// Add lifetime bound to lifetime parameter of trait
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a, 'b> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b> { }



// Add builtin bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltinBoundToTypeParameterOfTrait<T: Send> { }



// Add second type parameter to trait
#[cfg(cfail1)]
trait TraitAddSecondTypeParameterToTrait<T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondTypeParameterToTrait<T, S> { }



// Add second lifetime parameter to trait
#[cfg(cfail1)]
trait TraitAddSecondLifetimeParameterToTrait<'a> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeParameterToTrait<'a, 'b> { }



// Add second trait bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondTraitBoundToTypeParameterOfTrait<T: ReferencedTrait0 + ReferencedTrait1> { }



// Add second lifetime bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTrait<'a, 'b, T: 'a + 'b> { }



// Add second lifetime bound to lifetime parameter of trait
#[cfg(cfail1)]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b, 'b, 'c> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTrait<'a: 'b + 'c, 'b, 'c> { }



// Add second builtin bound to type parameter of trait
#[cfg(cfail1)]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTrait<T: Send + Sync> { }



struct ReferenceType0 {}
struct ReferenceType1 {}



// Add trait bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }



// Add lifetime bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToTypeParameterOfTraitWhere<'a, T> where T: 'a { }



// Add lifetime bound to lifetime parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b> where 'a: 'b { }



// Add builtin bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }



// Add second trait bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T> where T: ReferencedTrait0 { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondTraitBoundToTypeParameterOfTraitWhere<T>
    where T: ReferencedTrait0 + ReferencedTrait1 { }



// Add second lifetime bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeBoundToTypeParameterOfTraitWhere<'a, 'b, T> where T: 'a + 'b { }



// Add second lifetime bound to lifetime parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondLifetimeBoundToLifetimeParameterOfTraitWhere<'a, 'b, 'c> where 'a: 'b + 'c { }



// Add second builtin bound to type parameter of trait in where clause
#[cfg(cfail1)]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send { }

#[cfg(not(cfail1))]
#[rustc_dirty(label="hir_owner", cfg="cfail2")]
#[rustc_clean(label="hir_owner", cfg="cfail3")]
trait TraitAddSecondBuiltinBoundToTypeParameterOfTraitWhere<T> where T: Send + Sync { }


// Change return type of method indirectly by modifying a use statement
mod change_return_type_of_method_indirectly_use {
    #[cfg(cfail1)]
    use super::ReferenceType0 as ReturnType;
    #[cfg(not(cfail1))]
    use super::ReferenceType1 as ReturnType;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeReturnType {
        #[rustc_dirty(label="hir_owner", cfg="cfail2")]
        #[rustc_clean(label="hir_owner", cfg="cfail3")]
        fn method() -> ReturnType;
    }
}



// Change type of method parameter indirectly by modifying a use statement
mod change_method_parameter_type_indirectly_by_use {
    #[cfg(cfail1)]
    use super::ReferenceType0 as ArgType;
    #[cfg(not(cfail1))]
    use super::ReferenceType1 as ArgType;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeArgType {
        #[rustc_dirty(label="hir_owner", cfg="cfail2")]
        #[rustc_clean(label="hir_owner", cfg="cfail3")]
        fn method(a: ArgType);
    }
}



// Change trait bound of method type parameter indirectly by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use {
    #[cfg(cfail1)]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeBoundOfMethodTypeParameter {
        #[rustc_dirty(label="hir_owner", cfg="cfail2")]
        #[rustc_clean(label="hir_owner", cfg="cfail3")]
        fn method<T: Bound>(a: T);
    }
}



// Change trait bound of method type parameter in where clause indirectly
// by modifying a use statement
mod change_method_parameter_type_bound_indirectly_by_use_where {
    #[cfg(cfail1)]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_clean(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeBoundOfMethodTypeParameterWhere {
        #[rustc_dirty(label="hir_owner", cfg="cfail2")]
        #[rustc_clean(label="hir_owner", cfg="cfail3")]
        fn method<T>(a: T) where T: Bound;
    }
}



// Change trait bound of trait type parameter indirectly by modifying a use statement
mod change_method_type_parameter_bound_indirectly {
    #[cfg(cfail1)]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeTraitBound<T: Bound> {
        fn method(a: T);
    }
}



// Change trait bound of trait type parameter in where clause indirectly
// by modifying a use statement
mod change_method_type_parameter_bound_indirectly_where {
    #[cfg(cfail1)]
    use super::ReferencedTrait0 as Bound;
    #[cfg(not(cfail1))]
    use super::ReferencedTrait1 as Bound;

    #[rustc_dirty(label="hir_owner", cfg="cfail2")]
    #[rustc_clean(label="hir_owner", cfg="cfail3")]
    trait TraitChangeTraitBoundWhere<T> where T: Bound {
        fn method(a: T);
    }
}

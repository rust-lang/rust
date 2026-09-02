macro_rules! include_example {
    ($name:literal) => {
        concat!(
            "```rust,compile_fail\n",
            include_str!(concat!("../../../tests/ui/attributes/doc_examples/", $name, ".rs")),
            "```\n",
            "produces:\n",
            " ```text\n",
            include_str!(concat!("../../../tests/ui/attributes/doc_examples/", $name, ".stderr")),
            "```\n",
        )
    };
}

#[cfg_attr(not(bootstrap), doc(attribute = "rustc_dump_clauses"))]
/// Dumps the list of [`ty::Clause`]s as computed by the [`clauses_of`] query.
///
/// See [`AttributeKind::RustcDumpClauses`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_clauses")]
///
/// # Example: super trait bounds are not elaborated
///
#[doc = include_example!("rustc_dump_clauses_super_trait")]
///
/// [`clauses_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.clauses_of
/// [`ty::Clause`]: ../rustc_middle/ty/struct.Clause.html
const _: () = ();

#[doc(attribute = "rustc_dump_def_parents")]
/// Dumps the parents of the annotated item and of any anonymous constants contained within it.
///
/// See also [`opt_parent`](../rustc_middle/ty/struct.TyCtxt.html#method.opt_parent).
///
/// See [`AttributeKind::RustcDumpDefParents`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_def_parents")]
const _: () = ();

#[doc(attribute = "rustc_dump_def_path")]
/// Dumps the def path of the annotated item.
///
/// See also [`def_path_str`] and [`def_path_str_with_args`].
///
/// See [`AttributeKind::RustcDumpDefPath`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_def_path")]
///
/// [`def_path_str`]: ../rustc_middle/ty/struct.TyCtxt.html#method.def_path_str
/// [`def_path_str_with_args`]: ../rustc_middle/ty/struct.TyCtxt.html#method.def_path_str_with_args
const _: () = ();

#[doc(attribute = "rustc_dump_generics")]
/// Dumps the generics of the annotated item.
///
/// See [`generics_of`] and [`ty::Generics`] for what "generics" means here.
///
/// See [`AttributeKind::RustcDumpGenerics`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_generics")]
///
/// [`generics_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.generics_of
/// [`ty::Generics`]: ../rustc_middle/ty/struct.Generics.html
const _: () = ();

#[doc(attribute = "rustc_dump_hidden_type_of_opaques")]
/// Dumps the hidden types of the opaque items in this crate.
///
/// This ends up calling the [`type_of`] query, which, for opaque types, reveals their hidden types.
///
/// See [`AttributeKind::RustcDumpHiddenTypeOfOpaques`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_hidden_type_of_opaques")]
///
/// [`type_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.type_of
const _: () = ();

#[doc(attribute = "rustc_dump_inferred_outlives")]
/// Dumps the inferred outlives-clauses of the annotated item.
///
/// See also the [`inferred_outlives_of`] query.
///
/// See [`AttributeKind::RustcDumpInferredOutlives`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_inferred_outlives")]
///
/// [`inferred_outlives_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.inferred_outlives_of
const _: () = ();

#[doc(attribute = "rustc_dump_item_bounds")]
/// Dumps the item bounds of the annotated item.
///
/// This ends up calling the [`item_bounds`] query and prints the [`ty::Clause`] of the item.
///
/// See [`AttributeKind::RustcDumpItemBounds`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_item_bounds")]
///
/// [`item_bounds`]: ../rustc_middle/ty/struct.TyCtxt.html#method.item_bounds
/// [`ty::Clause`]: ../rustc_middle/ty/struct.Clause.html
const _: () = ();

#[doc(attribute = "rustc_dump_layout")]
/// Dumps the layout of the annotated item.
///
/// This ends up calling the [`layout_of`] query to get the [`Layout`] of the annotated item. If used
/// with the `debug` modifier, it will print the entirety of `Layout`. Other modifiers will print
/// only parts of it.
///
/// See [`AttributeKind::RustcDumpLayout`] for the internal representation of this attribute.
///
/// # Example: `debug`
///
#[doc = include_example!("rustc_dump_layout_debug")]
///
/// # Example: `largest_niche`
///
#[doc = include_example!("rustc_dump_layout_largest_niche")]
///
/// # Example: `size`
///
#[doc = include_example!("rustc_dump_layout_size")]
///
/// # Example: `align`
///
#[doc = include_example!("rustc_dump_layout_align")]
///
/// # Example: `backend_repr`
///
#[doc = include_example!("rustc_dump_layout_backend_repr")]
///
/// # Example: `homogeneous_aggregate`
///
#[doc = include_example!("rustc_dump_layout_homogeneous_aggregate")]
///
/// [`layout_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.layout_of
/// [`Layout`]: rustc_abi::Layout
const _: () = ();

#[doc(attribute = "rustc_dump_object_lifetime_defaults")]
/// Dumps the trait object lifetime defaults induced by the type parameters of the annotated item.
///
/// It will dump this information separately for each type parameter of the annotated item.
///
/// See also the [`object_lifetime_default`] query.
///
/// See [`AttributeKind::RustcDumpObjectLifetimeDefaults`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_object_lifetime_defaults")]
///
/// [`object_lifetime_default`]: ../rustc_middle/ty/struct.TyCtxt.html#method.object_lifetime_default
const _: () = ();

#[doc(attribute = "rustc_dump_symbol_name")]
/// Dumps the symbol name of the annotated item, also demangling it if necessary.
///
/// See also the [`symbol_name`] query.
///
/// See [`AttributeKind::RustcDumpSymbolName`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_symbol_name")]
///
/// [`symbol_name`]: ../rustc_middle/ty/struct.TyCtxt.html#method.symbol_name
const _: () = ();

#[doc(attribute = "rustc_dump_variances")]
/// Dumps the variances of the annotated item.
///
/// See also the [`variances_of`] query and [`ty::Variance`].
///
/// See [`AttributeKind::RustcDumpVariances`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_variances")]
///
/// [`variances_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.variances_of
/// [`ty::Variance`]: ../rustc_middle/ty/enum.Variance.html
const _: () = ();

#[doc(attribute = "rustc_dump_variances_of_opaques")]
/// Dumps the variances of opaque types in this crate.
///
/// See also the [`variances_of`] query.
///
/// See [`AttributeKind::RustcDumpVariancesOfOpaques`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_variances_of_opaques")]
///
/// [`variances_of`]: ../rustc_middle/ty/struct.TyCtxt.html#method.variances_of
const _: () = ();

#[doc(attribute = "rustc_dump_vtable")]
/// Dumps the virtual method table ("vtable") of the annotated item.
///
/// See also the [`vtable_entries`] query.
///
/// See [`AttributeKind::RustcDumpVtable`] for the internal representation of this attribute.
///
/// # Example
///
#[doc = include_example!("rustc_dump_vtable")]
///
/// [`vtable_entries`]: ../rustc_middle/ty/struct.TyCtxt.html#method.vtable_entries
const _: () = ();

#[cfg_attr(not(bootstrap), doc(attribute = "rustc_dump_clauses"))]
/// Dumps the clauses of the annotated item.
///
/// See [`AttributeKind::RustcDumpClauses`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(negative_impls)]
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_clauses]
/// fn function<T: Send>(_t: T) {}
///
/// #[rustc_dump_clauses]
/// trait Trait: Sync {
///     #[rustc_dump_clauses]
///     type Assoc;
/// }
///
/// #[rustc_dump_clauses]
/// struct Struct<T: ?Sized>(T);
///
/// #[rustc_dump_clauses]
/// impl<T: ?Sized> !Sync for Struct<T> {}
/// ```
///
/// ```text
/// error: rustc_dump_clauses
///  --> src/lib.rs:5:1
///   |
/// 5 | fn function<T: Send>(_t: T) {}
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^
///   |
///   = note: Binder { value: TraitClause(<T as std::marker::Sized>, polarity:Positive), bound_vars: [] }
///   = note: Binder { value: TraitClause(<T as std::marker::Send>, polarity:Positive), bound_vars: [] }
///
/// error: rustc_dump_clauses
///  --> src/lib.rs:8:1
///   |
/// 8 | trait Trait: Sync {
///   | ^^^^^^^^^^^^^^^^^
///   |
///   = note: Binder { value: TraitClause(<Self as std::marker::MetaSized>, polarity:Positive), bound_vars: [] }
///   = note: Binder { value: TraitClause(<Self as std::marker::Sync>, polarity:Positive), bound_vars: [] }
///   = note: Binder { value: TraitClause(<Self as Trait>, polarity:Positive), bound_vars: [] }
///
/// error: rustc_dump_clauses
///   --> src/lib.rs:14:1
///    |
/// 14 | struct Struct<T: ?Sized>(T);
///    | ^^^^^^^^^^^^^^^^^^^^^^^^
///    |
///    = note: Binder { value: TraitClause(<T as std::marker::MetaSized>, polarity:Positive), bound_vars: [] }
///
/// error: rustc_dump_clauses
///   --> src/lib.rs:17:1
///    |
/// 17 | impl<T: ?Sized> !Sync for Struct<T> {}
///    | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///    |
///    = note: Binder { value: TraitClause(<T as std::marker::MetaSized>, polarity:Positive), bound_vars: [] }
///
/// error: rustc_dump_clauses
///   --> src/lib.rs:10:5
///    |
/// 10 |     type Assoc;
///    |     ^^^^^^^^^^
///    |
///    = note: Binder { value: TraitClause(<Self as std::marker::MetaSized>, polarity:Positive), bound_vars: [] }
///    = note: Binder { value: TraitClause(<Self as std::marker::Sync>, polarity:Positive), bound_vars: [] }
///    = note: Binder { value: TraitClause(<Self as Trait>, polarity:Positive), bound_vars: [] }
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_def_parents")]
/// Dumps the def parents of the annotated item.
///
/// See [`AttributeKind::RustcDumpDefParents`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// fn parent() {
///     #[rustc_dump_def_parents]
///     fn child() {}
/// }
///
/// struct Struct<const N: usize>;
///
/// const CONST: Struct<42> = Struct::<
///     {
///         #[rustc_dump_def_parents]
///         fn baby() {}
///
///         42
///     },
/// >;
/// ```
///
/// ```text
/// error: rustc_dump_def_parents: DefId(0:4 ~ playground[05f0]::parent::child)
///   --> src/lib.rs:5:5
///    |
///  5 |     fn child() {}
///    |     ^^^^^^^^^^
///    |
/// note: DefId(0:3 ~ playground[05f0]::parent)
///   --> src/lib.rs:3:1
///    |
///  3 | fn parent() {
///    | ^^^^^^^^^^^
/// note: DefId(0:0 ~ playground[05f0])
///   --> src/lib.rs:1:1
///    |
///  1 | / #![feature(rustc_attrs)]
///  2 | |
///  3 | | fn parent() {
///  4 | |     #[rustc_dump_def_parents]
/// ...  |
/// 16 | |     },
/// 17 | | >;
///    | |__^
///
/// error: rustc_dump_def_parents: DefId(0:11 ~ playground[05f0]::CONST::{constant#1}::baby)
///   --> src/lib.rs:13:9
///    |
/// 13 |         fn baby() {}
///    |         ^^^^^^^^^
///    |
/// note: DefId(0:10 ~ playground[05f0]::CONST::{constant#1})
///   --> src/lib.rs:11:5
///    |
/// 11 | /     {
/// 12 | |         #[rustc_dump_def_parents]
/// 13 | |         fn baby() {}
/// ...  |
/// 16 | |     },
///    | |_____^
/// note: DefId(0:8 ~ playground[05f0]::CONST)
///   --> src/lib.rs:10:1
///    |
/// 10 | const CONST: Struct<42> = Struct::<
///    | ^^^^^^^^^^^^^^^^^^^^^^^
/// note: DefId(0:0 ~ playground[05f0])
///   --> src/lib.rs:1:1
///    |
///  1 | / #![feature(rustc_attrs)]
///  2 | |
///  3 | | fn parent() {
///  4 | |     #[rustc_dump_def_parents]
/// ...  |
/// 16 | |     },
/// 17 | | >;
///    | |__^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_def_path")]
/// Dumps the def path of the annotated items.
///
/// See [`AttributeKind::RustcDumpDefPath`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_def_path]
/// fn main() {
///     || {
///         unsafe extern "C" {
///             #[rustc_dump_def_path]
///             static Foo: u8;
///         }
///     };
/// }
///
/// mod a {
///     mod b {
///         mod c {
///             #[rustc_dump_def_path]
///             fn d() {}
///         }
///     }
/// }
/// ```
///
/// ```text
/// error: def-path(main)
///  --> src/main.rs:3:1
///   |
/// 3 | #[rustc_dump_def_path]
///   | ^^^^^^^^^^^^^^^^^^^^^^
///
/// error: def-path(a::b::c::d)
///   --> src/main.rs:16:13
///    |
/// 16 |             #[rustc_dump_def_path]
///    |             ^^^^^^^^^^^^^^^^^^^^^^
///
/// error: def-path(main::{closure#0}::Foo)
///  --> src/main.rs:7:13
///   |
/// 7 |             #[rustc_dump_def_path]
///   |             ^^^^^^^^^^^^^^^^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_generics")]
/// Dumps the generic of the annotated item.
///
/// See [`AttributeKind::RustcDumpGenerics`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_generics]
/// struct Struct<'lifetime, const CONST: usize, GENERIC> {
///     stuff: &'lifetime [GENERIC; CONST],
/// }
/// ```
///
/// ```text
/// error: rustc_dump_generics: DefId(0:3 ~ playground[05f0]::Struct)
///  --> src/lib.rs:4:1
///   |
/// 4 | struct Struct<'lifetime, const CONST: usize, GENERIC> {
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///   |
/// note: Generics {
///           parent: None,
///           parent_count: 0,
///           own_params: [
///               GenericParamDef {
///                   name: "'lifetime",
///                   def_id: DefId(0:4 ~ playground[05f0]::Struct::'lifetime),
///                   index: 0,
///                   pure_wrt_drop: false,
///                   kind: Lifetime,
///               },
///               GenericParamDef {
///                   name: "CONST",
///                   def_id: DefId(0:5 ~ playground[05f0]::Struct::CONST),
///                   index: 1,
///                   pure_wrt_drop: false,
///                   kind: Const {
///                       has_default: false,
///                   },
///               },
///               GenericParamDef {
///                   name: "GENERIC",
///                   def_id: DefId(0:6 ~ playground[05f0]::Struct::GENERIC),
///                   index: 2,
///                   pure_wrt_drop: false,
///                   kind: Type {
///                       has_default: false,
///                       synthetic: false,
///                   },
///               },
///           ],
///           param_def_id_to_index: [
///               (
///                   DefId(0:4 ~ playground[05f0]::Struct::'lifetime),
///                   0,
///               ),
///               (
///                   DefId(0:5 ~ playground[05f0]::Struct::CONST),
///                   1,
///               ),
///               (
///                   DefId(0:6 ~ playground[05f0]::Struct::GENERIC),
///                   2,
///               ),
///           ],
///           has_self: false,
///           has_late_bound_regions: None,
///       }
///  --> src/lib.rs:4:1
///   |
/// 4 | struct Struct<'lifetime, const CONST: usize, GENERIC> {
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_hidden_type_of_opaques")]
/// Dumps the hidden types of the opaque items in this crate.
///
/// See [`AttributeKind::RustcDumpHiddenTypeOfOpaques`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// // Use `-Zverbose-internals` for a more verbosive description
/// #![feature(rustc_attrs)]
/// #![rustc_dump_hidden_type_of_opaques]
///
/// trait Foo {
///     fn hello(&self) -> impl Sized;
/// }
///
/// fn hello<'s, T: Foo>(x: &'s T) -> impl Sized + use<'s, T> {
///     x.hello()
/// }
/// ```
///
/// ```text
/// error: impl Sized
///  --> src/lib.rs:8:35
///   |
/// 8 | fn hello<'s, T: Foo>(x: &'s T) -> impl Sized + use<'s, T> {
///   |                                   ^^^^^^^^^^^^^^^^^^^^^^^
/// ```
///
/// ```rust,compile_fail
/// #![feature(type_alias_impl_trait)]
/// #![feature(rustc_attrs)]
/// #![rustc_dump_hidden_type_of_opaques]
///
/// trait Trait {}
///
/// struct MyType;
///
/// impl Trait for MyType {}
///
/// type Alias = impl Trait;
///
/// #[define_opaque(Alias)]
/// fn new() -> Alias {
///     MyType
/// }
/// ```
///
/// ```text
/// error: MyType
///   --> src/lib.rs:11:14
///    |
/// 11 | type Alias = impl Trait;
///    |              ^^^^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_inferred_outlives")]
/// Dumps the inferred outlives-clauses of the annotated item.
///
/// See [`AttributeKind::RustcDumpInferredOutlives`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// trait Trait<'x, T> where T: 'x {
///     type Type;
/// }
///
/// #[rustc_dump_inferred_outlives]
/// struct Foo<'a, A, B> where A: Trait<'a, B>
/// {
///     foo: <A as Trait<'a, B>>::Type
/// }
/// ```
///
/// ```text
/// error: rustc_dump_inferred_outlives
///  --> src/lib.rs:8:1
///   |
/// 8 | struct Foo<'a, A, B> where A: Trait<'a, B>
///   | ^^^^^^^^^^^^^^^^^^^^
///   |
///   = note: B: 'a
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_item_bounds")]
/// Dumps the item bounds of the annotated item.
///
/// See [`AttributeKind::RustcDumpItemBounds`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// trait Trait<T> {
///     #[rustc_dump_item_bounds]
///     type Assoc: PartialEq<String>
/// }
/// ```
///
/// ```text
/// error: rustc_dump_item_bounds
///  --> src/lib.rs:5:5
///   |
/// 5 |     type Assoc: PartialEq<String>
///   |     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
///   |
///   = note: Binder { value: TraitClause(<<Self as Trait<T>>::Assoc as std::cmp::PartialEq<std::string::String>>, polarity:Positive), bound_vars: [] }
///   = note: Binder { value: TraitClause(<<Self as Trait<T>>::Assoc as std::marker::Sized>, polarity:Positive), bound_vars: [] }
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_layout")]
/// Dumps the layout of the annotated item.
///
/// There are various options available for dumping subsets of [`Layout`](rustc_abi::Layout).
///
/// See [`AttributeKind::RustcDumpLayout`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(portable_simd)]
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_layout(debug)]
/// pub union Union {
///     Float: f32,
///     Int: u32,
/// }
///
/// #[rustc_dump_layout(largest_niche)]
/// type Alias = Option<char>;
///
/// #[rustc_dump_layout(homogeneous_aggregate)]
/// #[repr(C)]
/// struct Struct {
///     field: [u8; 32],
///     unit: (),
/// }
///
/// #[rustc_dump_layout(backend_repr)]
/// type Simd = std::simd::u32x4;
///
/// #[rustc_dump_layout(align)]
/// enum Enum {
///     Bytes([u8; 4]),
///     Int(u32),
/// }
///
/// #[rustc_dump_layout(size)]
/// type ID = std::any::TypeId;
/// ```
///
/// ```text
/// error: layout_of(Union) = Layout {
///            size: Size(4 bytes),
///            align: AbiAlign {
///                abi: Align(4 bytes),
///            },
///            backend_repr: Memory {
///                sized: true,
///            },
///            fields: Union(
///                2,
///            ),
///            largest_niche: None,
///            uninhabited: false,
///            variants: Single {
///                index: 0,
///            },
///            max_repr_align: None,
///            unadjusted_abi_align: Align(4 bytes),
///            randomization_seed: 12593054327350107868,
///        }
///  --> src/lib.rs:5:1
///   |
/// 5 | pub union Union {
///   | ^^^^^^^^^^^^^^^
///
/// error: largest_niche: Some(Niche { offset: Size(0 bytes), value: u32, valid_range: (..=1114111) | (4294967295..) })
///   --> src/lib.rs:11:1
///    |
/// 11 | type Alias = Option<char>;
///    | ^^^^^^^^^^
///
/// error: homogeneous_aggregate: Ok(Homogeneous(Reg { kind: Integer, size: Size(1 bytes) }))
///   --> src/lib.rs:15:1
///    |
/// 15 | struct Struct {
///    | ^^^^^^^^^^^^^
///
/// error: backend_repr: SimdVector { element: u32 is .., count: BackendLaneCount(4) }
///   --> src/lib.rs:21:1
///    |
/// 21 | type Simd = std::simd::u32x4;
///    | ^^^^^^^^^
///
/// error: align: Align(4 bytes)
///   --> src/lib.rs:24:1
///    |
/// 24 | enum Enum {
///    | ^^^^^^^^^
///
/// error: size: Size(16 bytes)
///   --> src/lib.rs:30:1
///    |
/// 30 | type ID = std::any::TypeId;
///    | ^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_object_lifetime_defaults")]
/// Dumps the lifetime defaults of the annotated item.
///
/// See [`AttributeKind::RustcDumpObjectLifetimeDefaults`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_object_lifetime_defaults]
/// struct Ref<'a, T: 'a>(&'a T);
/// ```
///
/// ```text
/// error: 'a
///  --> src/lib.rs:4:16
///   |
/// 4 | struct Ref<'a, T: 'a>(&'a T);
///   |
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_symbol_name")]
/// Dumps the symbol name of the annotated item, also demangling it if necessary.
///
/// See [`AttributeKind::RustcDumpSymbolName`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_symbol_name]
/// fn mangled() {}
///
/// #[rustc_dump_symbol_name]
/// #[unsafe(no_mangle)]
/// fn no_mangle() {}
///
/// unsafe extern "C" {
///     #[rustc_dump_symbol_name]
///     fn abort();
/// }
/// ```
///
/// ```text
/// error: symbol-name(_RNvCsvCIBz2sTtb_10playground7mangled)
///  --> src/lib.rs:3:1
///   |
/// 3 | #[rustc_dump_symbol_name]
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// error: demangling(playground[5f0ee0a16bea70f]::mangled)
///  --> src/lib.rs:3:1
///   |
/// 3 | #[rustc_dump_symbol_name]
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// error: demangling-alt(playground::mangled)
///  --> src/lib.rs:3:1
///   |
/// 3 | #[rustc_dump_symbol_name]
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// error: symbol-name(no_mangle)
///  --> src/lib.rs:6:1
///   |
/// 6 | #[rustc_dump_symbol_name]
///   | ^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// error: symbol-name(abort)
///   --> src/lib.rs:11:5
///    |
/// 11 |     #[rustc_dump_symbol_name]
///    |     ^^^^^^^^^^^^^^^^^^^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_variances")]
/// Dumps the variances of the annotated item.
///
/// See also [`Variance`](../rustc_middle/ty/enum.Variance.html).
///
/// See [`AttributeKind::RustcDumpVariances`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_variances]
/// struct Ref<'a, T> {
///     r: &'a T,
/// }
///
/// #[rustc_dump_variances]
/// struct RefMut<'a, T> {
///     r: &'a mut T,
/// }
///
/// #[rustc_dump_variances]
/// struct CellRef<'a, T> {
///     r: &'a core::cell::UnsafeCell<T>,
/// }
///
/// #[rustc_dump_variances]
/// fn x<T, U>(_t: T) -> U {
///     todo!()
/// }
/// ```
///
/// ```text
/// error: ['a: +, T: +]
///  --> src/lib.rs:4:1
///   |
/// 4 | struct Ref<'a, T> {
///   | ^^^^^^^^^^^^^^^^^
///
/// error: ['a: +, T: o]
///  --> src/lib.rs:9:1
///   |
/// 9 | struct RefMut<'a, T> {
///   | ^^^^^^^^^^^^^^^^^^^^
///
/// error: ['a: +, T: o]
///   --> src/lib.rs:14:1
///    |
/// 14 | struct CellRef<'a, T> {
///    | ^^^^^^^^^^^^^^^^^^^^^
///
/// error: [T: -, U: +]
///   --> src/lib.rs:19:1
///    |
/// 19 | fn x<T, U>(_t: T) -> U {
///    | ^^^^^^^^^^^^^^^^^^^^^^
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_variances_of_opaques")]
/// Dumps the variances of opaque types in this crate.
///
/// See [`AttributeKind::RustcDumpVariancesOfOpaques`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
/// #![rustc_dump_variances_of_opaques]
///
/// trait Captures<'a> {}
/// impl<T> Captures<'_> for T {}
///
/// fn not_captured_early<'a: 'a>() -> impl Sized {}
///
/// fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {}
///
/// fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}
///
/// fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {}
/// ```
///
/// ```text
/// error: ['a: *, 'a: o]
///  --> src/lib.rs:7:36
///   |
/// 7 | fn not_captured_early<'a: 'a>() -> impl Sized {}
///   |                                    ^^^^^^^^^^
///
/// error: ['a: *, 'a: o]
///  --> src/lib.rs:9:32
///   |
/// 9 | fn captured_early<'a: 'a>() -> impl Sized + Captures<'a> {}
///   |                                ^^^^^^^^^^^^^^^^^^^^^^^^^
///
/// error: ['a: o]
///   --> src/lib.rs:11:40
///    |
/// 11 | fn not_captured_late<'a>(_: &'a ()) -> impl Sized {}
///    |                                        ^^^^^^^^^^
///
/// error: ['a: o]
///   --> src/lib.rs:13:36
///    |
/// 13 | fn captured_late<'a>(_: &'a ()) -> impl Sized + Captures<'a> {}
///    |
/// ```
const _: () = ();

#[doc(attribute = "rustc_dump_vtable")]
/// Dumps the virtual method table ("vtable") of the annotated item.
///
/// See [`AttributeKind::RustcDumpVtable`] for the internal representation of this attribute.
///
/// # Example
///
/// ```rust,compile_fail
/// #![feature(rustc_attrs)]
///
/// #[rustc_dump_vtable]
/// type X = dyn Send;
///
/// #[rustc_dump_vtable]
/// type Y = dyn core::any::Any;
///
/// struct C;
///
/// #[rustc_dump_vtable]
/// impl Iterator for C {
///     type Item = ();
///     fn next(&mut self) -> Option<Self::Item> {
///         Some(())
///     }
/// }
/// ```
///
/// ```text
/// error: vtable entries: [
///            MetadataDropInPlace,
///            MetadataSize,
///            MetadataAlign,
///        ]
///  --> src/lib.rs:4:1
///   |
/// 4 | type X = dyn Send;
///   | ^^^^^^
///
/// error: vtable entries: [
///            MetadataDropInPlace,
///            MetadataSize,
///            MetadataAlign,
///            Method(<dyn Any as Any>::type_id - shim(reify)),
///        ]
///  --> src/lib.rs:7:1
///   |
/// 7 | type Y = dyn core::any::Any;
///   | ^^^^^^
///
/// error: vtable entries: [
///            MetadataDropInPlace,
///            MetadataSize,
///            MetadataAlign,
///            Method(<C as Iterator>::next),
///            Method(<C as Iterator>::size_hint),
///            Method(<C as Iterator>::advance_by),
///            Method(<C as Iterator>::nth),
///        ]
///   --> src/lib.rs:12:1
///    |
/// 12 | impl Iterator for C {
///    | ^^^^^^^^^^^^^^^^^^^
/// ```
const _: () = ();

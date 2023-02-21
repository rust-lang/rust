// For more information about type metadata and type metadata identifiers for cross-language LLVM
// CFI support, see Type metadata in the design document in the tracking issue #89653.

// FIXME(rcvalle): Identify C char and integer type uses and encode them with their respective
// builtin type encodings as specified by the Itanium C++ ABI for extern function types with the "C"
// calling convention to use this encoding for cross-language LLVM CFI.

use bitflags::bitflags;
use core::fmt::Display;
use rustc_data_structures::base_n;
use rustc_data_structures::fx::FxHashMap;
use rustc_hir as hir;
use rustc_middle::ty::subst::{GenericArg, GenericArgKind, SubstsRef};
use rustc_middle::ty::{
    self, Const, ExistentialPredicate, FloatTy, FnSig, IntTy, List, Region, RegionKind, TermKind,
    Ty, TyCtxt, UintTy,
};
use rustc_span::def_id::DefId;
use rustc_span::symbol::sym;
use rustc_target::abi::call::{Conv, FnAbi};
use rustc_target::spec::abi::Abi;
use std::fmt::Write as _;

/// Type and extended type qualifiers.
#[derive(Eq, Hash, PartialEq)]
enum TyQ {
    None,
    Const,
    Mut,
}

/// Substitution dictionary key.
#[derive(Eq, Hash, PartialEq)]
enum DictKey<'tcx> {
    Ty(Ty<'tcx>, TyQ),
    Region(Region<'tcx>),
    Const(Const<'tcx>),
    Predicate(ExistentialPredicate<'tcx>),
}

bitflags! {
    /// Options for typeid_for_fnabi and typeid_for_fnsig.
    pub struct TypeIdOptions: u32 {
        const NO_OPTIONS = 0;
        const GENERALIZE_POINTERS = 1;
        const GENERALIZE_REPR_C = 2;
    }
}

/// Options for encode_ty.
type EncodeTyOptions = TypeIdOptions;

/// Options for transform_ty.
type TransformTyOptions = TypeIdOptions;

/// Converts a number to a disambiguator (see
/// <https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html>).
fn to_disambiguator(num: u64) -> String {
    if let Some(num) = num.checked_sub(1) {
        format!("s{}_", base_n::encode(num as u128, 62))
    } else {
        "s_".to_string()
    }
}

/// Converts a number to a sequence number (see
/// <https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangle.seq-id>).
fn to_seq_id(num: usize) -> String {
    if let Some(num) = num.checked_sub(1) {
        base_n::encode(num as u128, 36).to_uppercase()
    } else {
        "".to_string()
    }
}

/// Substitutes a component if found in the substitution dictionary (see
/// <https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-compression>).
fn compress<'tcx>(
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    key: DictKey<'tcx>,
    comp: &mut String,
) {
    match dict.get(&key) {
        Some(num) => {
            comp.clear();
            let _ = write!(comp, "S{}_", to_seq_id(*num));
        }
        None => {
            dict.insert(key, dict.len());
        }
    }
}

// FIXME(rcvalle): Move to compiler/rustc_middle/src/ty/sty.rs after C types work is done, possibly
// along with other is_c_type methods.
/// Returns whether a `ty::Ty` is `c_void`.
fn is_c_void_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> bool {
    match ty.kind() {
        ty::Adt(adt_def, ..) => {
            let def_id = adt_def.0.did;
            let crate_name = tcx.crate_name(def_id.krate);
            tcx.item_name(def_id).as_str() == "c_void"
                && (crate_name == sym::core || crate_name == sym::std || crate_name == sym::libc)
        }
        _ => false,
    }
}

/// Encodes a const using the Itanium C++ ABI as a literal argument (see
/// <https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling.literal>).
fn encode_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    c: Const<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: EncodeTyOptions,
) -> String {
    // L<element-type>[n]<element-value>E as literal argument
    let mut s = String::from('L');

    // Element type
    s.push_str(&encode_ty(tcx, c.ty(), dict, options));

    // The only allowed types of const parameters are bool, u8, u16, u32, u64, u128, usize i8, i16,
    // i32, i64, i128, isize, and char. The bool value false is encoded as 0 and true as 1.
    fn push_signed_value<T: Display + PartialOrd>(s: &mut String, value: T, zero: T) {
        if value < zero {
            s.push('n')
        };
        let _ = write!(s, "{value}");
    }

    fn push_unsigned_value<T: Display>(s: &mut String, value: T) {
        let _ = write!(s, "{value}");
    }

    if let Some(scalar_int) = c.kind().try_to_scalar_int() {
        let signed = c.ty().is_signed();
        match scalar_int.size().bits() {
            8 if signed => push_signed_value(&mut s, scalar_int.try_to_i8().unwrap(), 0),
            16 if signed => push_signed_value(&mut s, scalar_int.try_to_i16().unwrap(), 0),
            32 if signed => push_signed_value(&mut s, scalar_int.try_to_i32().unwrap(), 0),
            64 if signed => push_signed_value(&mut s, scalar_int.try_to_i64().unwrap(), 0),
            128 if signed => push_signed_value(&mut s, scalar_int.try_to_i128().unwrap(), 0),
            8 => push_unsigned_value(&mut s, scalar_int.try_to_u8().unwrap()),
            16 => push_unsigned_value(&mut s, scalar_int.try_to_u16().unwrap()),
            32 => push_unsigned_value(&mut s, scalar_int.try_to_u32().unwrap()),
            64 => push_unsigned_value(&mut s, scalar_int.try_to_u64().unwrap()),
            128 => push_unsigned_value(&mut s, scalar_int.try_to_u128().unwrap()),
            _ => {
                bug!("encode_const: unexpected size `{:?}`", scalar_int.size().bits());
            }
        };
    } else {
        bug!("encode_const: unexpected type `{:?}`", c.ty());
    }

    // Close the "L..E" pair
    s.push('E');

    compress(dict, DictKey::Const(c), &mut s);

    s
}

/// Encodes a FnSig using the Itanium C++ ABI with vendor extended type qualifiers and types for
/// Rust types that are not used at the FFI boundary.
#[instrument(level = "trace", skip(tcx, dict))]
fn encode_fnsig<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_sig: &FnSig<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: TypeIdOptions,
) -> String {
    // Function types are delimited by an "F..E" pair
    let mut s = String::from("F");

    let mut encode_ty_options = EncodeTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("encode_fnsig: invalid option(s) `{:?}`", options.bits()));
    match fn_sig.abi {
        Abi::C { .. } => {
            encode_ty_options.insert(EncodeTyOptions::GENERALIZE_REPR_C);
        }
        _ => {
            encode_ty_options.remove(EncodeTyOptions::GENERALIZE_REPR_C);
        }
    }

    // Encode the return type
    let transform_ty_options = TransformTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("encode_fnsig: invalid option(s) `{:?}`", options.bits()));
    let ty = transform_ty(tcx, fn_sig.output(), transform_ty_options);
    s.push_str(&encode_ty(tcx, ty, dict, encode_ty_options));

    // Encode the parameter types
    let tys = fn_sig.inputs();
    if !tys.is_empty() {
        for ty in tys {
            let ty = transform_ty(tcx, *ty, transform_ty_options);
            s.push_str(&encode_ty(tcx, ty, dict, encode_ty_options));
        }

        if fn_sig.c_variadic {
            s.push('z');
        }
    } else {
        if fn_sig.c_variadic {
            s.push('z');
        } else {
            // Empty parameter lists, whether declared as () or conventionally as (void), are
            // encoded with a void parameter specifier "v".
            s.push('v')
        }
    }

    // Close the "F..E" pair
    s.push('E');

    s
}

/// Encodes a predicate using the Itanium C++ ABI with vendor extended type qualifiers and types for
/// Rust types that are not used at the FFI boundary.
fn encode_predicate<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicate: ty::PolyExistentialPredicate<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: EncodeTyOptions,
) -> String {
    // u<length><name>[I<element-type1..element-typeN>E], where <element-type> is <subst>, as vendor
    // extended type.
    let mut s = String::new();
    match predicate.as_ref().skip_binder() {
        ty::ExistentialPredicate::Trait(trait_ref) => {
            let name = encode_ty_name(tcx, trait_ref.def_id);
            let _ = write!(s, "u{}{}", name.len(), &name);
            s.push_str(&encode_substs(tcx, trait_ref.substs, dict, options));
        }
        ty::ExistentialPredicate::Projection(projection) => {
            let name = encode_ty_name(tcx, projection.def_id);
            let _ = write!(s, "u{}{}", name.len(), &name);
            s.push_str(&encode_substs(tcx, projection.substs, dict, options));
            match projection.term.unpack() {
                TermKind::Ty(ty) => s.push_str(&encode_ty(tcx, ty, dict, options)),
                TermKind::Const(c) => s.push_str(&encode_const(tcx, c, dict, options)),
            }
        }
        ty::ExistentialPredicate::AutoTrait(def_id) => {
            let name = encode_ty_name(tcx, *def_id);
            let _ = write!(s, "u{}{}", name.len(), &name);
        }
    };
    compress(dict, DictKey::Predicate(*predicate.as_ref().skip_binder()), &mut s);
    s
}

/// Encodes predicates using the Itanium C++ ABI with vendor extended type qualifiers and types for
/// Rust types that are not used at the FFI boundary.
fn encode_predicates<'tcx>(
    tcx: TyCtxt<'tcx>,
    predicates: &List<ty::PolyExistentialPredicate<'tcx>>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: EncodeTyOptions,
) -> String {
    // <predicate1[..predicateN]>E as part of vendor extended type
    let mut s = String::new();
    let predicates: Vec<ty::PolyExistentialPredicate<'tcx>> = predicates.iter().collect();
    for predicate in predicates {
        s.push_str(&encode_predicate(tcx, predicate, dict, options));
    }
    s
}

/// Encodes a region using the Itanium C++ ABI as a vendor extended type.
fn encode_region<'tcx>(
    _tcx: TyCtxt<'tcx>,
    region: Region<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    _options: EncodeTyOptions,
) -> String {
    // u6region[I[<region-disambiguator>][<region-index>]E] as vendor extended type
    let mut s = String::new();
    match region.kind() {
        RegionKind::ReLateBound(debruijn, r) => {
            s.push_str("u6regionI");
            // Debruijn index, which identifies the binder, as region disambiguator
            let num = debruijn.index() as u64;
            if num > 0 {
                s.push_str(&to_disambiguator(num));
            }
            // Index within the binder
            let _ = write!(s, "{}", r.var.index() as u64);
            s.push('E');
            compress(dict, DictKey::Region(region), &mut s);
        }
        RegionKind::ReErased => {
            s.push_str("u6region");
            compress(dict, DictKey::Region(region), &mut s);
        }
        RegionKind::ReEarlyBound(..)
        | RegionKind::ReFree(..)
        | RegionKind::ReStatic
        | RegionKind::ReError(_)
        | RegionKind::ReVar(..)
        | RegionKind::RePlaceholder(..) => {
            bug!("encode_region: unexpected `{:?}`", region.kind());
        }
    }
    s
}

/// Encodes substs using the Itanium C++ ABI with vendor extended type qualifiers and types for Rust
/// types that are not used at the FFI boundary.
fn encode_substs<'tcx>(
    tcx: TyCtxt<'tcx>,
    substs: SubstsRef<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: EncodeTyOptions,
) -> String {
    // [I<subst1..substN>E] as part of vendor extended type
    let mut s = String::new();
    let substs: Vec<GenericArg<'_>> = substs.iter().collect();
    if !substs.is_empty() {
        s.push('I');
        for subst in substs {
            match subst.unpack() {
                GenericArgKind::Lifetime(region) => {
                    s.push_str(&encode_region(tcx, region, dict, options));
                }
                GenericArgKind::Type(ty) => {
                    s.push_str(&encode_ty(tcx, ty, dict, options));
                }
                GenericArgKind::Const(c) => {
                    s.push_str(&encode_const(tcx, c, dict, options));
                }
            }
        }
        s.push('E');
    }
    s
}

/// Encodes a ty:Ty name, including its crate and path disambiguators and names.
fn encode_ty_name(tcx: TyCtxt<'_>, def_id: DefId) -> String {
    // Encode <name> for use in u<length><name>[I<element-type1..element-typeN>E], where
    // <element-type> is <subst>, using v0's <path> without v0's extended form of paths:
    //
    // N<namespace-tagN>..N<namespace-tag1>
    // C<crate-disambiguator><crate-name>
    // <path-disambiguator1><path-name1>..<path-disambiguatorN><path-nameN>
    //
    // With additional tags for DefPathData::Impl and DefPathData::ForeignMod. For instance:
    //
    //     pub type Type1 = impl Send;
    //     let _: Type1 = <Struct1<i32>>::foo;
    //     fn foo1(_: Type1) { }
    //
    //     pub type Type2 = impl Send;
    //     let _: Type2 = <Trait1<i32>>::foo;
    //     fn foo2(_: Type2) { }
    //
    //     pub type Type3 = impl Send;
    //     let _: Type3 = <i32 as Trait1<i32>>::foo;
    //     fn foo3(_: Type3) { }
    //
    //     pub type Type4 = impl Send;
    //     let _: Type4 = <Struct1<i32> as Trait1<i32>>::foo;
    //     fn foo3(_: Type4) { }
    //
    // Are encoded as:
    //
    //     _ZTSFvu29NvNIC1234_5crate8{{impl}}3fooIu3i32EE
    //     _ZTSFvu27NvNtC1234_5crate6Trait13fooIu3dynIu21NtC1234_5crate6Trait1Iu3i32Eu6regionES_EE
    //     _ZTSFvu27NvNtC1234_5crate6Trait13fooIu3i32S_EE
    //     _ZTSFvu27NvNtC1234_5crate6Trait13fooIu22NtC1234_5crate7Struct1Iu3i32ES_EE
    //
    // The reason for not using v0's extended form of paths is to use a consistent and simpler
    // encoding, as the reasoning for using it isn't relevand for type metadata identifiers (i.e.,
    // keep symbol names close to how methods are represented in error messages). See
    // https://rust-lang.github.io/rfcs/2603-rust-symbol-name-mangling-v0.html#methods.
    let mut s = String::new();

    // Start and namespace tags
    let mut def_path = tcx.def_path(def_id);
    def_path.data.reverse();
    for disambiguated_data in &def_path.data {
        s.push('N');
        s.push_str(match disambiguated_data.data {
            hir::definitions::DefPathData::Impl => "I", // Not specified in v0's <namespace>
            hir::definitions::DefPathData::ForeignMod => "F", // Not specified in v0's <namespace>
            hir::definitions::DefPathData::TypeNs(..) => "t",
            hir::definitions::DefPathData::ValueNs(..) => "v",
            hir::definitions::DefPathData::ClosureExpr => "C",
            hir::definitions::DefPathData::Ctor => "c",
            hir::definitions::DefPathData::AnonConst => "k",
            hir::definitions::DefPathData::ImplTrait => "i",
            hir::definitions::DefPathData::CrateRoot
            | hir::definitions::DefPathData::Use
            | hir::definitions::DefPathData::GlobalAsm
            | hir::definitions::DefPathData::ImplTraitAssocTy
            | hir::definitions::DefPathData::MacroNs(..)
            | hir::definitions::DefPathData::LifetimeNs(..) => {
                bug!("encode_ty_name: unexpected `{:?}`", disambiguated_data.data);
            }
        });
    }

    // Crate disambiguator and name
    s.push('C');
    s.push_str(&to_disambiguator(tcx.stable_crate_id(def_path.krate).to_u64()));
    let crate_name = tcx.crate_name(def_path.krate).to_string();
    let _ = write!(s, "{}{}", crate_name.len(), &crate_name);

    // Disambiguators and names
    def_path.data.reverse();
    for disambiguated_data in &def_path.data {
        let num = disambiguated_data.disambiguator as u64;
        if num > 0 {
            s.push_str(&to_disambiguator(num));
        }

        let name = disambiguated_data.data.to_string();
        let _ = write!(s, "{}", name.len());

        // Prepend a '_' if name starts with a digit or '_'
        if let Some(first) = name.as_bytes().get(0) {
            if first.is_ascii_digit() || *first == b'_' {
                s.push('_');
            }
        } else {
            bug!("encode_ty_name: invalid name `{:?}`", name);
        }

        s.push_str(&name);
    }

    s
}

/// Encodes a ty:Ty using the Itanium C++ ABI with vendor extended type qualifiers and types for
/// Rust types that are not used at the FFI boundary.
fn encode_ty<'tcx>(
    tcx: TyCtxt<'tcx>,
    ty: Ty<'tcx>,
    dict: &mut FxHashMap<DictKey<'tcx>, usize>,
    options: EncodeTyOptions,
) -> String {
    let mut typeid = String::new();

    match ty.kind() {
        // Primitive types
        ty::Bool => {
            typeid.push('b');
        }

        ty::Int(..) | ty::Uint(..) | ty::Float(..) => {
            // u<length><type-name> as vendor extended type
            let mut s = String::from(match ty.kind() {
                ty::Int(IntTy::I8) => "u2i8",
                ty::Int(IntTy::I16) => "u3i16",
                ty::Int(IntTy::I32) => "u3i32",
                ty::Int(IntTy::I64) => "u3i64",
                ty::Int(IntTy::I128) => "u4i128",
                ty::Int(IntTy::Isize) => "u5isize",
                ty::Uint(UintTy::U8) => "u2u8",
                ty::Uint(UintTy::U16) => "u3u16",
                ty::Uint(UintTy::U32) => "u3u32",
                ty::Uint(UintTy::U64) => "u3u64",
                ty::Uint(UintTy::U128) => "u4u128",
                ty::Uint(UintTy::Usize) => "u5usize",
                ty::Float(FloatTy::F32) => "u3f32",
                ty::Float(FloatTy::F64) => "u3f64",
                _ => "",
            });
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::Char => {
            // u4char as vendor extended type
            let mut s = String::from("u4char");
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::Str => {
            // u3str as vendor extended type
            let mut s = String::from("u3str");
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::Never => {
            // u5never as vendor extended type
            let mut s = String::from("u5never");
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // Compound types
        // () in Rust is equivalent to void return type in C
        _ if ty.is_unit() => {
            typeid.push('v');
        }

        // Sequence types
        ty::Tuple(tys) => {
            // u5tupleI<element-type1..element-typeN>E as vendor extended type
            let mut s = String::from("u5tupleI");
            for ty in tys.iter() {
                s.push_str(&encode_ty(tcx, ty, dict, options));
            }
            s.push('E');
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::Array(ty0, len) => {
            // A<array-length><element-type>
            let mut s = String::from("A");
            let _ = write!(s, "{}", &len.kind().try_to_scalar().unwrap().to_u64().unwrap());
            s.push_str(&encode_ty(tcx, *ty0, dict, options));
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::Slice(ty0) => {
            // u5sliceI<element-type>E as vendor extended type
            let mut s = String::from("u5sliceI");
            s.push_str(&encode_ty(tcx, *ty0, dict, options));
            s.push('E');
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // User-defined types
        ty::Adt(adt_def, substs) => {
            let mut s = String::new();
            let def_id = adt_def.0.did;
            if options.contains(EncodeTyOptions::GENERALIZE_REPR_C) && adt_def.repr().c() {
                // For cross-language CFI support, the encoding must be compatible at the FFI
                // boundary. For instance:
                //
                //     struct type1 {};
                //     void foo(struct type1* bar) {}
                //
                // Is encoded as:
                //
                //     _ZTSFvP5type1E
                //
                // So, encode any repr(C) user-defined type for extern function types with the "C"
                // calling convention (or extern types [i.e., ty::Foreign]) as <length><name>, where
                // <name> is <unscoped-name>.
                let name = tcx.item_name(def_id).to_string();
                let _ = write!(s, "{}{}", name.len(), &name);
                compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            } else {
                // u<length><name>[I<element-type1..element-typeN>E], where <element-type> is
                // <subst>, as vendor extended type.
                let name = encode_ty_name(tcx, def_id);
                let _ = write!(s, "u{}{}", name.len(), &name);
                s.push_str(&encode_substs(tcx, substs, dict, options));
                compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            }
            typeid.push_str(&s);
        }

        ty::Foreign(def_id) => {
            // <length><name>, where <name> is <unscoped-name>
            let mut s = String::new();
            let name = tcx.item_name(*def_id).to_string();
            let _ = write!(s, "{}{}", name.len(), &name);
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // Function types
        ty::FnDef(def_id, substs)
        | ty::Closure(def_id, substs)
        | ty::Generator(def_id, substs, ..) => {
            // u<length><name>[I<element-type1..element-typeN>E], where <element-type> is <subst>,
            // as vendor extended type.
            let mut s = String::new();
            let name = encode_ty_name(tcx, *def_id);
            let _ = write!(s, "u{}{}", name.len(), &name);
            s.push_str(&encode_substs(tcx, substs, dict, options));
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // Pointer types
        ty::Ref(region, ty0, ..) => {
            // [U3mut]u3refI<element-type>E as vendor extended type qualifier and type
            let mut s = String::new();
            s.push_str("u3refI");
            s.push_str(&encode_ty(tcx, *ty0, dict, options));
            s.push('E');
            compress(dict, DictKey::Ty(tcx.mk_imm_ref(*region, *ty0), TyQ::None), &mut s);
            if ty.is_mutable_ptr() {
                s = format!("{}{}", "U3mut", &s);
                compress(dict, DictKey::Ty(ty, TyQ::Mut), &mut s);
            }
            typeid.push_str(&s);
        }

        ty::RawPtr(tm) => {
            // P[K]<element-type>
            let mut s = String::new();
            s.push_str(&encode_ty(tcx, tm.ty, dict, options));
            if !ty.is_mutable_ptr() {
                s = format!("{}{}", "K", &s);
                compress(dict, DictKey::Ty(tm.ty, TyQ::Const), &mut s);
            };
            s = format!("{}{}", "P", &s);
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        ty::FnPtr(fn_sig) => {
            // PF<return-type><parameter-type1..parameter-typeN>E
            let mut s = String::from("P");
            s.push_str(&encode_fnsig(tcx, &fn_sig.skip_binder(), dict, TypeIdOptions::NO_OPTIONS));
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // Trait types
        ty::Dynamic(predicates, region, kind) => {
            // u3dynI<element-type1[..element-typeN]>E, where <element-type> is <predicate>, as
            // vendor extended type.
            let mut s = String::from(match kind {
                ty::Dyn => "u3dynI",
                ty::DynStar => "u7dynstarI",
            });
            s.push_str(&encode_predicates(tcx, predicates, dict, options));
            s.push_str(&encode_region(tcx, *region, dict, options));
            s.push('E');
            compress(dict, DictKey::Ty(ty, TyQ::None), &mut s);
            typeid.push_str(&s);
        }

        // Unexpected types
        ty::Bound(..)
        | ty::Error(..)
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::Infer(..)
        | ty::Alias(..)
        | ty::Param(..)
        | ty::Placeholder(..) => {
            bug!("encode_ty: unexpected `{:?}`", ty.kind());
        }
    };

    typeid
}

// Transforms a ty:Ty for being encoded and used in the substitution dictionary. It transforms all
// c_void types into unit types unconditionally, and generalizes all pointers if
// TransformTyOptions::GENERALIZE_POINTERS option is set.
#[instrument(level = "trace", skip(tcx))]
fn transform_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>, options: TransformTyOptions) -> Ty<'tcx> {
    let mut ty = ty;

    match ty.kind() {
        ty::Bool
        | ty::Int(..)
        | ty::Uint(..)
        | ty::Float(..)
        | ty::Char
        | ty::Str
        | ty::Never
        | ty::Foreign(..)
        | ty::Dynamic(..) => {}

        _ if ty.is_unit() => {}

        ty::Tuple(tys) => {
            ty = tcx.mk_tup(tys.iter().map(|ty| transform_ty(tcx, ty, options)));
        }

        ty::Array(ty0, len) => {
            let len = len.kind().try_to_scalar().unwrap().to_u64().unwrap();
            ty = tcx.mk_array(transform_ty(tcx, *ty0, options), len);
        }

        ty::Slice(ty0) => {
            ty = tcx.mk_slice(transform_ty(tcx, *ty0, options));
        }

        ty::Adt(adt_def, substs) => {
            if is_c_void_ty(tcx, ty) {
                ty = tcx.mk_unit();
            } else if options.contains(TransformTyOptions::GENERALIZE_REPR_C) && adt_def.repr().c()
            {
                ty = tcx.mk_adt(*adt_def, ty::List::empty());
            } else if adt_def.repr().transparent() && adt_def.is_struct() {
                let variant = adt_def.non_enum_variant();
                let param_env = tcx.param_env(variant.def_id);
                let field = variant.fields.iter().find(|field| {
                    let ty = tcx.type_of(field.did).subst_identity();
                    let is_zst =
                        tcx.layout_of(param_env.and(ty)).map_or(false, |layout| layout.is_zst());
                    !is_zst
                });
                if let Some(field) = field {
                    let ty0 = tcx.type_of(field.did).subst(tcx, substs);
                    // Generalize any repr(transparent) user-defined type that is either a pointer
                    // or reference, and either references itself or any other type that contains or
                    // references itself, to avoid a reference cycle.
                    if ty0.is_any_ptr() && ty0.contains(ty) {
                        ty = transform_ty(
                            tcx,
                            ty0,
                            options | TransformTyOptions::GENERALIZE_POINTERS,
                        );
                    } else {
                        ty = transform_ty(tcx, ty0, options);
                    }
                } else {
                    // Transform repr(transparent) types without non-ZST field into ()
                    ty = tcx.mk_unit();
                }
            } else {
                ty = tcx.mk_adt(*adt_def, transform_substs(tcx, substs, options));
            }
        }

        ty::FnDef(def_id, substs) => {
            ty = tcx.mk_fn_def(*def_id, transform_substs(tcx, substs, options));
        }

        ty::Closure(def_id, substs) => {
            ty = tcx.mk_closure(*def_id, transform_substs(tcx, substs, options));
        }

        ty::Generator(def_id, substs, movability) => {
            ty = tcx.mk_generator(*def_id, transform_substs(tcx, substs, options), *movability);
        }

        ty::Ref(region, ty0, ..) => {
            if options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                if ty.is_mutable_ptr() {
                    ty = tcx.mk_mut_ref(tcx.lifetimes.re_static, tcx.mk_unit());
                } else {
                    ty = tcx.mk_imm_ref(tcx.lifetimes.re_static, tcx.mk_unit());
                }
            } else {
                if ty.is_mutable_ptr() {
                    ty = tcx.mk_mut_ref(*region, transform_ty(tcx, *ty0, options));
                } else {
                    ty = tcx.mk_imm_ref(*region, transform_ty(tcx, *ty0, options));
                }
            }
        }

        ty::RawPtr(tm) => {
            if options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                if ty.is_mutable_ptr() {
                    ty = tcx.mk_mut_ptr(tcx.mk_unit());
                } else {
                    ty = tcx.mk_imm_ptr(tcx.mk_unit());
                }
            } else {
                if ty.is_mutable_ptr() {
                    ty = tcx.mk_mut_ptr(transform_ty(tcx, tm.ty, options));
                } else {
                    ty = tcx.mk_imm_ptr(transform_ty(tcx, tm.ty, options));
                }
            }
        }

        ty::FnPtr(fn_sig) => {
            if options.contains(TransformTyOptions::GENERALIZE_POINTERS) {
                ty = tcx.mk_imm_ptr(tcx.mk_unit());
            } else {
                let parameters: Vec<Ty<'tcx>> = fn_sig
                    .skip_binder()
                    .inputs()
                    .iter()
                    .map(|ty| transform_ty(tcx, *ty, options))
                    .collect();
                let output = transform_ty(tcx, fn_sig.skip_binder().output(), options);
                ty = tcx.mk_fn_ptr(ty::Binder::bind_with_vars(
                    tcx.mk_fn_sig(
                        parameters,
                        output,
                        fn_sig.c_variadic(),
                        fn_sig.unsafety(),
                        fn_sig.abi(),
                    ),
                    fn_sig.bound_vars(),
                ));
            }
        }

        ty::Bound(..)
        | ty::Error(..)
        | ty::GeneratorWitness(..)
        | ty::GeneratorWitnessMIR(..)
        | ty::Infer(..)
        | ty::Alias(..)
        | ty::Param(..)
        | ty::Placeholder(..) => {
            bug!("transform_ty: unexpected `{:?}`", ty.kind());
        }
    }

    ty
}

/// Transforms substs for being encoded and used in the substitution dictionary.
fn transform_substs<'tcx>(
    tcx: TyCtxt<'tcx>,
    substs: SubstsRef<'tcx>,
    options: TransformTyOptions,
) -> SubstsRef<'tcx> {
    let substs = substs.iter().map(|subst| {
        if let GenericArgKind::Type(ty) = subst.unpack() {
            if is_c_void_ty(tcx, ty) {
                tcx.mk_unit().into()
            } else {
                transform_ty(tcx, ty, options).into()
            }
        } else {
            subst
        }
    });
    tcx.mk_substs(substs)
}

/// Returns a type metadata identifier for the specified FnAbi using the Itanium C++ ABI with vendor
/// extended type qualifiers and types for Rust types that are not used at the FFI boundary.
#[instrument(level = "trace", skip(tcx))]
pub fn typeid_for_fnabi<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_abi: &FnAbi<'tcx, Ty<'tcx>>,
    options: TypeIdOptions,
) -> String {
    // A name is mangled by prefixing "_Z" to an encoding of its name, and in the case of functions
    // its type.
    let mut typeid = String::from("_Z");

    // Clang uses the Itanium C++ ABI's virtual tables and RTTI typeinfo structure name as type
    // metadata identifiers for function pointers. The typeinfo name encoding is a two-character
    // code (i.e., 'TS') prefixed to the type encoding for the function.
    typeid.push_str("TS");

    // Function types are delimited by an "F..E" pair
    typeid.push('F');

    // A dictionary of substitution candidates used for compression (see
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-compression).
    let mut dict: FxHashMap<DictKey<'tcx>, usize> = FxHashMap::default();

    let mut encode_ty_options = EncodeTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("typeid_for_fnabi: invalid option(s) `{:?}`", options.bits()));
    match fn_abi.conv {
        Conv::C => {
            encode_ty_options.insert(EncodeTyOptions::GENERALIZE_REPR_C);
        }
        _ => {
            encode_ty_options.remove(EncodeTyOptions::GENERALIZE_REPR_C);
        }
    }

    // Encode the return type
    let transform_ty_options = TransformTyOptions::from_bits(options.bits())
        .unwrap_or_else(|| bug!("typeid_for_fnabi: invalid option(s) `{:?}`", options.bits()));
    let ty = transform_ty(tcx, fn_abi.ret.layout.ty, transform_ty_options);
    typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));

    // Encode the parameter types
    if !fn_abi.c_variadic {
        if !fn_abi.args.is_empty() {
            for arg in fn_abi.args.iter() {
                let ty = transform_ty(tcx, arg.layout.ty, transform_ty_options);
                typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));
            }
        } else {
            // Empty parameter lists, whether declared as () or conventionally as (void), are
            // encoded with a void parameter specifier "v".
            typeid.push('v');
        }
    } else {
        for n in 0..fn_abi.fixed_count as usize {
            let ty = transform_ty(tcx, fn_abi.args[n].layout.ty, transform_ty_options);
            typeid.push_str(&encode_ty(tcx, ty, &mut dict, encode_ty_options));
        }

        typeid.push('z');
    }

    // Close the "F..E" pair
    typeid.push('E');

    typeid
}

/// Returns a type metadata identifier for the specified FnSig using the Itanium C++ ABI with vendor
/// extended type qualifiers and types for Rust types that are not used at the FFI boundary.
pub fn typeid_for_fnsig<'tcx>(
    tcx: TyCtxt<'tcx>,
    fn_sig: &FnSig<'tcx>,
    options: TypeIdOptions,
) -> String {
    // A name is mangled by prefixing "_Z" to an encoding of its name, and in the case of functions
    // its type.
    let mut typeid = String::from("_Z");

    // Clang uses the Itanium C++ ABI's virtual tables and RTTI typeinfo structure name as type
    // metadata identifiers for function pointers. The typeinfo name encoding is a two-character
    // code (i.e., 'TS') prefixed to the type encoding for the function.
    typeid.push_str("TS");

    // A dictionary of substitution candidates used for compression (see
    // https://itanium-cxx-abi.github.io/cxx-abi/abi.html#mangling-compression).
    let mut dict: FxHashMap<DictKey<'tcx>, usize> = FxHashMap::default();

    // Encode the function signature
    typeid.push_str(&encode_fnsig(tcx, fn_sig, &mut dict, options));

    typeid
}

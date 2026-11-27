//! Function pointer type discrimination for pointer authentication.

//! This module implements Rust's equivalent of Clang's function pointer type
//! discriminator computation used in pointer authentication.
//!
//! Compatibility with Clang is a primary goal. The discriminator produced for a
//! given external "C" function type must match the value computed by Clang so that
//! function pointers can be exchanged safely between Rust and C code while
//! preserving pointer authentication semantics.
//!
//! The implementation mirrors Clang's behavior in
//! `ASTContext::encodeTypeForFunctionPointerAuth`, ensuring that identical
//! C-compatible function types produce identical discriminators. See:
//! <https://clang.llvm.org/doxygen/ASTContext_8cpp.html#abb1375e068e807917527842d05cadea3>.
//!
//! ## Overview
//!
//! The computation is structured into three conceptual stages:
//!
//! ### 1. Type normalization and lowering
//!    Rust types are converted into a language-independent representation
//!    (`ClangDiscTy`) that mirrors the type categories used by Clang when computing
//!    function pointer discriminators. This includes canonicalization such as
//!    treating all pointer-like types uniformly and mapping Rust constructs onto
//!    their closest C equivalents.
//!    One notable exception is C `_Complex`. Rust has no corresponding native type,
//!    so there is no canonical Rust representation to map onto Clang's `_Complex`
//!    type category. Rather than infer one (for example, by treating `(f32, f32)`
//!    or `(f64, f64)` as complex numbers), this implementation leaves such
//!    representation choices to users and does not provide dedicated `_Complex`
//!    encoding.
//!
//! ### 2. Type encoding
//!    The lowered representation is serialized into a byte stream using rules
//!    intended to match Clang's implementation in:
//!    `encodeTypeForFunctionPointerAuth`. The resulting encoding describes the
//!    function signature in a target-independent form suitable for hashing.
//!
//! ### 3. Discriminator hashing
//!    The encoded byte stream is hashed using LLVM's stable SipHash-2-4 based
//!    discriminator algorithm. The implementation here is a direct translation
//!    of LLVM/Clang's logic and must remain bit-for-bit compatible. See:
//!    <https://github.com/llvm/llvm-project/blob/main/third-party/siphash/include/siphash/SipHash.h>.
//!    Defined in `llvm_siphash.rs`.
//!
//! ## Module structure
//!
//! - High-level API
//!   - `FnPtrDiscriminatorSource`
//!   - `ptrauth_compute_fn_ptr_type_discriminator_for`
//!   - `is_ptrauth_signed_abi`
//!   - `ptrauth_signing_discriminator_for`
//!
//! - Low-level API
//!   - `FnPtrTypeDiscriminatorInput` - canonical function signature input for
//!     discriminator calculation (as a convenience it exposes the function ABI
//!     through `abi()`)
//!   - `compute_fn_ptr_type_discriminator`
//!
//! - Clang-compatible type model
//!   - `ClangDiscTy`
//!   - `canonicalize_c_type`
//!   - `to_clang_disc_ty`
//!
//! - Encoding
//!   - `PtrauthEncoder`
//!   - `encode_ty`
//!
//! ## Compatibility requirements
//!
//! Any changes to the encoding or hashing logic should be validated against Clang's
//! discriminator computation. Divergence from Clang will result in incompatible
//! pointer authentication values across language boundaries.
//!
//! This implementation intentionally approximates Clang's behavior for
//! `extern "C"` and `extern "system"` function types only. It does NOT attempt
//! to model full Rust type system rules.

use rustc_abi::ExternAbi;
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, Unnormalized};
use rustc_session::{PointerAuthDiscrimination, PointerAuthSchema};
use rustc_span::sym;

use crate::ptrauth::llvm_siphash::llvm_pointer_auth_stable_siphash;

/// Types that can serve as a source for function pointer type discrimination.
///
/// This trait abstracts over the different compiler representations from which
/// a function signature can be obtained. Implementations construct the
/// canonical `FnPtrTypeDiscriminatorInput` consumed by the discriminator
/// computation.
///
/// This is intended primarily for ergonomic use at call sites, allowing code
/// to compute discriminators directly from an `Instance`, `Ty`, or `FnSig`
/// without manually constructing the intermediate representation.
pub trait FnPtrDiscriminatorSource<'tcx>: Sized {
    fn discriminator_input(self, tcx: TyCtxt<'tcx>) -> Option<FnPtrTypeDiscriminatorInput<'tcx>>;
}

/// Enables discriminator computation for function pointer types.
///
/// Only `FnPtr` is accepted: function items (`FnDef`) have no runtime code
/// address, and wrappers such as `Option<fn()>` must be handled separately.
impl<'tcx> FnPtrDiscriminatorSource<'tcx> for Ty<'tcx> {
    fn discriminator_input(self, tcx: TyCtxt<'tcx>) -> Option<FnPtrTypeDiscriminatorInput<'tcx>> {
        let ty = tcx.normalize_erasing_regions(
            ty::TypingEnv::fully_monomorphized(),
            Unnormalized::new(self),
        );
        match ty.kind() {
            ty::FnPtr(sig, header) => {
                Some(FnPtrTypeDiscriminatorInput::from_sig_tys(sig.skip_binder(), header))
            }
            _ => None,
        }
    }
}
/// Enables discriminator computation directly from monomorphized function
/// instances.
///
/// The instance's signature is instantiated using its generic arguments and
/// normalized before constructing the canonical discriminator input.
impl<'tcx> FnPtrDiscriminatorSource<'tcx> for Instance<'tcx> {
    fn discriminator_input(self, tcx: TyCtxt<'tcx>) -> Option<FnPtrTypeDiscriminatorInput<'tcx>> {
        let sig = tcx
            .instantiate_and_normalize_erasing_regions(
                self.args,
                ty::TypingEnv::fully_monomorphized(),
                tcx.fn_sig(self.def_id()),
            )
            .skip_binder();

        Some(FnPtrTypeDiscriminatorInput::from_sig(sig))
    }
}
/// Enables discriminator computation directly from instantiated function
/// signatures.
///
/// The signature is assumed to already be instantiated and normalized.
impl<'tcx> FnPtrDiscriminatorSource<'tcx> for ty::FnSig<'tcx> {
    fn discriminator_input(self, _: TyCtxt<'tcx>) -> Option<FnPtrTypeDiscriminatorInput<'tcx>> {
        Some(FnPtrTypeDiscriminatorInput::from_sig(self))
    }
}

/// Computes the function pointer type discriminator directly from a supported
/// source.
///
/// This is a convenience wrapper around
/// `FnPtrDiscriminatorSource::discriminator_input` and
/// `compute_fn_ptr_type_discriminator`.
///
/// Returns `None` if the supplied source does not represent a function pointer
/// type (for example, a non-function `Ty`).
pub fn ptrauth_compute_fn_ptr_type_discriminator_for<'tcx, S>(
    tcx: TyCtxt<'tcx>,
    source: S,
) -> Option<u16>
where
    S: FnPtrDiscriminatorSource<'tcx>,
{
    let input = source.discriminator_input(tcx)?;
    compute_fn_ptr_type_discriminator(tcx, &input)
}

/// Return true if a function pointer with this ABI is signed under pointer
/// authentication and calls through them authenticated.
pub fn is_ptrauth_signed_abi(abi: ExternAbi) -> bool {
    matches!(abi, ExternAbi::C { .. } | ExternAbi::System { .. })
}

/// Returns a discriminator for function pointer in `source` that is used for
/// signing and authentication under given `schema`. None if the function ABI
/// does not support function pointer authentication.
///
/// With function pointer type discrimination this is the discriminator
/// calculated for the `source`, otherwise `schema`'s constant discriminator.
pub fn ptrauth_signing_discriminator_for<'tcx, S>(
    tcx: TyCtxt<'tcx>,
    schema: &PointerAuthSchema,
    source: S,
) -> Option<u16>
where
    S: FnPtrDiscriminatorSource<'tcx>,
{
    let input = source.discriminator_input(tcx)?;
    if !is_ptrauth_signed_abi(input.abi()) {
        return None;
    }
    match schema.discrimination_kind {
        PointerAuthDiscrimination::Type => compute_fn_ptr_type_discriminator(tcx, &input),
        _ => Some(schema.constant_discriminator),
    }
}

/// Canonical representation of a function signature used for pointer
/// authentication discriminator generation.
#[derive(Debug)]
pub struct FnPtrTypeDiscriminatorInput<'tcx> {
    inputs: &'tcx [Ty<'tcx>],
    output: Ty<'tcx>,
    abi: ExternAbi,
    c_variadic: bool,
}

impl<'tcx> FnPtrTypeDiscriminatorInput<'tcx> {
    pub fn abi(&self) -> ExternAbi {
        self.abi
    }

    fn from_sig(sig: ty::FnSig<'tcx>) -> Self {
        FnPtrTypeDiscriminatorInput {
            inputs: sig.inputs(),
            output: sig.output(),
            abi: sig.abi(),
            c_variadic: sig.c_variadic(),
        }
    }

    fn from_sig_tys(sig: ty::FnSigTys<TyCtxt<'tcx>>, header: &ty::FnHeader<TyCtxt<'tcx>>) -> Self {
        FnPtrTypeDiscriminatorInput {
            inputs: sig.inputs(),
            output: sig.output(),
            abi: header.abi(),
            c_variadic: header.c_variadic(),
        }
    }
}

/// Computes the Clang-compatible function pointer type discriminator.
///
/// This is the low-level discriminator calculation function operating on
/// `FnPtrTypeDiscriminatorInput`. Returns `None` for unsupported ABIs (see
/// `is_ptrauth_signed_abi`).
fn compute_fn_ptr_type_discriminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &FnPtrTypeDiscriminatorInput<'tcx>,
) -> Option<u16> {
    if !is_ptrauth_signed_abi(input.abi) {
        return None;
    }

    let mut enc = PtrauthEncoder::new();
    enc.push(b'F');

    encode_ty(&mut enc, tcx, input.output);

    for &arg in input.inputs {
        encode_ty(&mut enc, tcx, arg);
    }

    if input.c_variadic {
        enc.push(b'z');
    }

    enc.push(b'E');

    Some(enc.finish())
}

// Clang disc type.
#[derive(Debug)]
enum ClangDiscTy<'tcx> {
    Int,
    Float(&'tcx ty::FloatTy),
    Bool,
    Char,

    // Pointer-like types in the C ABI sense.
    // This includes:
    // - raw pointers (`*const T`, `*mut T`)
    // - Rust references (`&T`, `&mut T`)
    // - function pointers
    // All collapse to a single Clang-compatible 'P' node.
    Pointer,

    Array { elem: Ty<'tcx> },

    // FIXME(jchlands) Decide if to support Complex types in future. Clang has
    // dedicated node for this `Type::Complex`, Rust does not. So we could match
    // against a Tuple(FP_TYPE, FP_TYPE).
    // Complex(Ty<'tcx>),
    Vector { bytes: u64 },

    EnumLikeInt,
    AdtName(String),
    Opaque,
    Void,
}

// Canonicalize Option-wrapped pointer types used to model C nullable pointers.
//
// Rust and Clang should compute identical discriminators for equivalent C APIs.
// Clang does not distinguish nullable from non-nullable pointer types when
// computing function pointer authentication discriminators, so
// `Option<fn>` and `Option<*mut T>` are encoded identically to their
// underlying pointer types.
//
// Although `Option<*mut T>` is not considered FFI-safe by Rust and triggers the
// `improper_ctypes`/`improper_ctypes_definitions` lints, this is a warning
// rather than a hard error. Canonicalizing it here preserves Clang-compatible
// discriminator computation.
//
// Please see the following tests for sample use cases:
// pauth-fn-ptr-type-discrimination-option-callback.rs,
// pauth-fn-ptr-type-discrimination-option-return.rs and pauth-fn-ptr-type-discrimination-option.rs
fn canonicalize_c_type<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> Ty<'tcx> {
    if let ty::Adt(def, args) = ty.kind()
        && tcx.is_diagnostic_item(sym::Option, def.did())
    {
        let inner = args.type_at(0);

        match inner.kind() {
            ty::FnPtr(..) | ty::RawPtr(..) => return inner,
            _ => {}
        }
    }

    ty
}

/// Lowers a Rust type into a Clang-compatible discriminator type.
///
/// This is not a full semantic translation of Rust types. It is a lossy mapping
/// that intentionally matches Clang's function pointer authentication encoding
/// rules where Rust has a direct language-level equivalent.
///
/// In particular C `_Complex`, without a canonical Rust equivalent, is not
/// recognized. This avoids introducing heuristics for user-defined
/// representations that may vary across codebases.
///
/// Important invariants:
/// - All pointer-like types (Rust refs, raw pointers, fn pointers) collapse to
///   `Pointer`.
/// - Struct/union types are encoded using name only, not layout.
/// - Enums are treated as integers.
/// - SIMD types are encoded only by total byte size (no lane semantics).
/// - No attempt is made to recognize user-defined representations of C
///   `_Complex` types.
/// This must remain in sync with Clang's `encodeTypeForFunctionPointerAuth`.
fn to_clang_disc_ty<'tcx>(tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) -> ClangDiscTy<'tcx> {
    let ty = canonicalize_c_type(tcx, ty);
    match ty.kind() {
        // C void / Rust ()
        _ if ty.is_unit() => ClangDiscTy::Void,

        // scalars
        ty::Bool => ClangDiscTy::Bool,
        ty::Char => ClangDiscTy::Char,

        ty::Int(_) | ty::Uint(_) => ClangDiscTy::Int,
        ty::Float(f) => ClangDiscTy::Float(f),

        // everything pointer-like collapses
        ty::RawPtr(..) | ty::Ref(..) | ty::FnPtr(..) | ty::Dynamic(..) | ty::Slice(_) | ty::Str => {
            ClangDiscTy::Pointer
        }

        // arrays ignore size
        ty::Array(elem, _) => ClangDiscTy::Array { elem: *elem },

        // enums to integer collapse
        ty::Adt(def, _) if def.is_enum() => ClangDiscTy::EnumLikeInt,
        // simd vectors
        ty::Adt(def, args) if def.repr().simd() => {
            // Clang encodes SIMD vectors by their total size
            let input = ty::PseudoCanonicalInput {
                typing_env: ty::TypingEnv::fully_monomorphized(),
                value: ty,
            };

            let Ok(layout) = tcx.layout_of(input) else {
                tcx.dcx().delayed_bug("could not compute SIMD layout");
                return ClangDiscTy::Opaque;
            };

            let bytes = layout.size.bytes();

            ClangDiscTy::Vector { bytes }
        }
        // structs/unions to name-based identity
        ty::Adt(def, _) => {
            let name = tcx.item_name(def.did()).to_string();
            ClangDiscTy::AdtName(name)
        }

        ty::Foreign(_) => ClangDiscTy::Opaque,

        _ => ClangDiscTy::Opaque,
    }
}

// Encoder
struct PtrauthEncoder {
    buf: Vec<u8>,
}

impl PtrauthEncoder {
    fn new() -> Self {
        Self { buf: Vec::new() }
    }

    fn push(&mut self, b: u8) {
        self.buf.push(b);
    }

    fn push_str(&mut self, s: &str) {
        self.buf.extend_from_slice(s.as_bytes());
    }

    fn finish(&self) -> u16 {
        llvm_pointer_auth_stable_siphash(&self.buf)
    }
}

/// Encodes a ClangDiscTy into the discriminator byte stream.
///
/// This format is intended to be bit-for-bit compatible with Clang's
/// `encodeTypeForFunctionPointerAuth`.
fn encode_ty<'tcx>(enc: &mut PtrauthEncoder, tcx: TyCtxt<'tcx>, ty: Ty<'tcx>) {
    let cty = to_clang_disc_ty(tcx, ty);

    match cty {
        // scalars
        ClangDiscTy::Bool | ClangDiscTy::Char | ClangDiscTy::Int => enc.push(b'i'),

        ClangDiscTy::Float(f) => match f.bit_width() {
            16 => enc.push_str("Dh"),
            32 => enc.push(b'f'),
            64 => enc.push(b'd'),
            128 => enc.push(b'g'),
            _ => enc.push(b'?'),
        },

        ClangDiscTy::Void => enc.push(b'v'),

        // pointer boundary (NO RECURSION)
        ClangDiscTy::Pointer => enc.push(b'P'),

        // arrays ignore size
        ClangDiscTy::Array { elem } => {
            enc.push(b'A');
            encode_ty(enc, tcx, elem);
        }

        // enums collapse
        ClangDiscTy::EnumLikeInt => enc.push(b'i'),

        // ADT identity
        ClangDiscTy::AdtName(name) => {
            enc.push_str(&name.len().to_string());
            enc.push_str(&name);
        }

        ClangDiscTy::Opaque => enc.push(b'?'),

        ClangDiscTy::Vector { bytes } => {
            enc.push_str("Dv");
            enc.push_str(&bytes.to_string());
        }
    }
}

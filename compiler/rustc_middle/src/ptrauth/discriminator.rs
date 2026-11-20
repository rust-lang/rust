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
//!
//!    Rust types are first canonicalized according to ABI representation
//!    (treating ABI-equivalent wrappers and niche representations as their
//!    underlying representation, and treating pointer-like types uniformly),
//!    then converted into a language-independent representation (`ClangDiscTy`)
//!    that mirrors the type categories used by Clang when computing function
//!    pointer discriminators.
//!
//!    One notable exception is C `_Complex`. Rust has no corresponding native type,
//!    so there is no canonical Rust representation to map onto Clang's `_Complex`
//!    type category. Rather than infer one (for example, by treating `(f32, f32)`
//!    or `(f64, f64)` as complex numbers), this implementation leaves such
//!    representation choices to users and does not provide dedicated `_Complex`
//!    encoding.
//!
//! ### 2. Type encoding
//!
//!    The lowered representation is serialized into a byte stream using rules
//!    intended to match Clang's implementation in:
//!    `encodeTypeForFunctionPointerAuth`. The resulting encoding describes the
//!    function signature in a target-independent form suitable for hashing.
//!
//! ### 3. Discriminator hashing
//!
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
//!   - `canonicalize_abi_compatible_type`
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

use rustc_abi::{ExternAbi, TagEncoding, Variants};
use rustc_middle::ty::{self, Instance, Ty, TyCtxt, Unnormalized};
use rustc_session::{PointerAuthDiscrimination, PointerAuthSchema};
use rustc_target::spec::Arch;

use crate::ptrauth::llvm_siphash::llvm_pointer_auth_stable_siphash;
use crate::ty::layout::LayoutCx;

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

    pub fn from_sig(sig: ty::FnSig<'tcx>) -> Self {
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

/// Encodes the signature described by `input` the way Clang's
/// `encodeTypeForFunctionPointerAuth` does, as the input to the discriminator
/// hash. Returns `None` for unsupported ABIs (see `is_ptrauth_signed_abi`).
fn encode_fn_ptr_type_discriminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &FnPtrTypeDiscriminatorInput<'tcx>,
) -> Option<PtrauthEncoder> {
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

    Some(enc)
}

/// Computes the Clang-compatible function pointer type discriminator.
///
/// This is the low-level discriminator calculation function operating on
/// `FnPtrTypeDiscriminatorInput`. Returns `None` for unsupported ABIs (see
/// `is_ptrauth_signed_abi`).
pub fn compute_fn_ptr_type_discriminator<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &FnPtrTypeDiscriminatorInput<'tcx>,
) -> Option<u16> {
    encode_fn_ptr_type_discriminator(tcx, input).map(|enc| enc.finish())
}

/// Returns the raw Clang-compatible type encoding used as input to the
/// discriminator hash.
///
/// This is primarily used for debugging and comparing Rust's encoding against
/// Clang's `encodeTypeForFunctionPointerAuth`.
pub fn debug_encode_fn_ptr_type<'tcx>(
    tcx: TyCtxt<'tcx>,
    input: &FnPtrTypeDiscriminatorInput<'tcx>,
) -> String {
    encode_fn_ptr_type_discriminator(tcx, input).map(|enc| enc.debug_string()).unwrap_or_default()
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
    // - `Box<T, Global>` (canonicalized to a raw pointer)
    // - `NonNull<T>` (canonicalized through its transparent representation)
    // - DST pointer-like types (`dyn Trait`, slices, `str`)
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

/// Canonicalizes Rust types that have the same ABI representation as a simpler
/// C-compatible type for discriminator purposes.
///
/// As a result ABI-compatible types get the same discriminator:
/// - a pattern type becomes its base type;
/// - `Box<T, Global>` becomes `*const T`;
/// - a size-0, alignment-1 type becomes `()`;
/// - a `repr(transparent)` wrapper becomes its non-1-ZST field;
/// - a two-variant enum with a niche becomes its payload.
///
/// The work is done on layouts, so it also merges some types that Rust does not
/// guarantee to be ABI-compatible, for example: `Option<bool>` and `bool`. That
/// only makes discriminators coarser, while still correct.
///
/// The steps repeat until nothing changes, as one can enable another:
/// `Option<Box<T>>` becomes `Box<T>`, then `*const T`.
fn canonicalize_abi_compatible_type<'tcx>(tcx: TyCtxt<'tcx>, mut ty: Ty<'tcx>) -> Ty<'tcx> {
    let typing_env = ty::TypingEnv::fully_monomorphized();

    loop {
        let before = ty;

        if let ty::Pat(base, _) = ty.kind() {
            ty = *base;
        }

        if ty.is_box_global(tcx) {
            ty = Ty::new_imm_ptr(
                tcx,
                ty.boxed_ty().expect("is_box_global() returned true, so this must be a Box"),
            );
        }

        if let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty)) {
            if layout.is_zst() && layout.align.bytes() == 1 {
                ty = tcx.types.unit;
            } else if matches!(ty.kind(), ty::Adt(..)) {
                let cx = LayoutCx::new(tcx, typing_env);
                ty = layout.peel_transparent_wrappers_from_non_1zst(&cx).ty;
            }
        }

        if let ty::Adt(def, args) = ty.kind()
            && def.is_enum()
            && def.variants().len() == 2
            && let Ok(layout) = tcx.layout_of(typing_env.as_query_input(ty))
            && let Variants::Multiple {
                tag_encoding: TagEncoding::Niche { untagged_variant, .. },
                ..
            } = layout.variants
        {
            let variant = def.variant(untagged_variant);
            if let [field] = &variant.fields.raw[..] {
                ty = tcx.normalize_erasing_regions(typing_env, field.ty(tcx, *args));
            }
        }

        if ty == before {
            return ty;
        }
    }
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
    let ty = canonicalize_abi_compatible_type(tcx, ty);
    match ty.kind() {
        // C void / Rust ()
        _ if ty.is_unit() || ty.is_never() => ClangDiscTy::Void,

        // scalars
        ty::Bool => ClangDiscTy::Bool,
        ty::Char => ClangDiscTy::Char,

        ty::Int(_) | ty::Uint(_) => ClangDiscTy::Int,
        ty::Float(f) => ClangDiscTy::Float(f),

        // Pointer-like types form a discriminator boundary: the pointee type
        // does not participate in the encoding.
        ty::RawPtr(..) | ty::Ref(..) | ty::FnPtr(..) | ty::Dynamic(..) | ty::Slice(_) | ty::Str => {
            ClangDiscTy::Pointer
        }

        // arrays ignore size
        ty::Array(elem, _) => ClangDiscTy::Array { elem: *elem },

        // enums to integer collapse - mirrors Clang's Type::Enum handling,
        // which recurses into the enum's underlying integer type.
        // A non-niche, data-carrying enum (e.g. Option<*mut T>) is not an
        // "enumerated type" in the C11 sense, such enums fall through to the
        // general Adt(_) => AdtName(..) arm below instead.
        ty::Adt(def, _) if def.is_enum() && def.is_payloadfree() => ClangDiscTy::EnumLikeInt,

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

    fn debug_string(&self) -> String {
        String::from_utf8_lossy(&self.buf).into_owned()
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
            16 => enc.push_str("DF16_"),
            32 => enc.push(b'f'),
            64 => enc.push(b'd'),
            128 => match tcx.sess.target.arch {
                Arch::AArch64
                    if !tcx.sess.target.is_like_darwin && !tcx.sess.target.is_like_windows =>
                {
                    enc.push(b'e')
                }
                _ => enc.push(b'g'),
            },
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

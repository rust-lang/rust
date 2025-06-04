use std::{fmt, iter};

use rustc_abi::{
    AddressSpace, Align, BackendRepr, CanonAbi, ExternAbi, HasDataLayout, Primitive, Reg, RegKind,
    Scalar, Size, TyAbiInterface, TyAndLayout,
};
use rustc_macros::HashStable_Generic;

pub use crate::spec::AbiMap;
use crate::spec::{HasTargetSpec, HasWasmCAbiOpt, HasX86AbiOpt, RustcAbi, WasmCAbi};

mod aarch64;
mod amdgpu;
mod arm;
mod avr;
mod bpf;
mod csky;
mod hexagon;
mod loongarch;
mod m68k;
mod mips;
mod mips64;
mod msp430;
mod nvptx64;
mod powerpc;
mod powerpc64;
mod riscv;
mod s390x;
mod sparc;
mod sparc64;
mod wasm;
mod x86;
mod x86_64;
mod x86_win32;
mod x86_win64;
mod xtensa;

#[derive(Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum PassMode {
    /// Ignore the argument.
    ///
    /// The argument is a ZST.
    Ignore,
    /// Pass the argument directly.
    ///
    /// The argument has a layout abi of `Scalar` or `Vector`.
    /// Unfortunately due to past mistakes, in rare cases on wasm, it can also be `Aggregate`.
    /// This is bad since it leaks LLVM implementation details into the ABI.
    /// (Also see <https://github.com/rust-lang/rust/issues/115666>.)
    Direct(ArgAttributes),
    /// Pass a pair's elements directly in two arguments.
    ///
    /// The argument has a layout abi of `ScalarPair`.
    Pair(ArgAttributes, ArgAttributes),
    /// Pass the argument after casting it. See the `CastTarget` docs for details.
    ///
    /// `pad_i32` indicates if a `Reg::i32()` dummy argument is emitted before the real argument.
    Cast { pad_i32: bool, cast: Box<CastTarget> },
    /// Pass the argument indirectly via a hidden pointer.
    ///
    /// The `meta_attrs` value, if any, is for the metadata (vtable or length) of an unsized
    /// argument. (This is the only mode that supports unsized arguments.)
    ///
    /// `on_stack` defines that the value should be passed at a fixed stack offset in accordance to
    /// the ABI rather than passed using a pointer. This corresponds to the `byval` LLVM argument
    /// attribute. The `byval` argument will use a byte array with the same size as the Rust type
    /// (which ensures that padding is preserved and that we do not rely on LLVM's struct layout),
    /// and will use the alignment specified in `attrs.pointee_align` (if `Some`) or the type's
    /// alignment (if `None`). This means that the alignment will not always
    /// match the Rust type's alignment; see documentation of `pass_by_stack_offset` for more info.
    ///
    /// `on_stack` cannot be true for unsized arguments, i.e., when `meta_attrs` is `Some`.
    Indirect { attrs: ArgAttributes, meta_attrs: Option<ArgAttributes>, on_stack: bool },
}

impl PassMode {
    /// Checks if these two `PassMode` are equal enough to be considered "the same for all
    /// function call ABIs". However, the `Layout` can also impact ABI decisions,
    /// so that needs to be compared as well!
    pub fn eq_abi(&self, other: &Self) -> bool {
        match (self, other) {
            (PassMode::Ignore, PassMode::Ignore) => true,
            (PassMode::Direct(a1), PassMode::Direct(a2)) => a1.eq_abi(a2),
            (PassMode::Pair(a1, b1), PassMode::Pair(a2, b2)) => a1.eq_abi(a2) && b1.eq_abi(b2),
            (
                PassMode::Cast { cast: c1, pad_i32: pad1 },
                PassMode::Cast { cast: c2, pad_i32: pad2 },
            ) => c1.eq_abi(c2) && pad1 == pad2,
            (
                PassMode::Indirect { attrs: a1, meta_attrs: None, on_stack: s1 },
                PassMode::Indirect { attrs: a2, meta_attrs: None, on_stack: s2 },
            ) => a1.eq_abi(a2) && s1 == s2,
            (
                PassMode::Indirect { attrs: a1, meta_attrs: Some(e1), on_stack: s1 },
                PassMode::Indirect { attrs: a2, meta_attrs: Some(e2), on_stack: s2 },
            ) => a1.eq_abi(a2) && e1.eq_abi(e2) && s1 == s2,
            _ => false,
        }
    }
}

// Hack to disable non_upper_case_globals only for the bitflags! and not for the rest
// of this module
pub use attr_impl::ArgAttribute;

#[allow(non_upper_case_globals)]
#[allow(unused)]
mod attr_impl {
    use rustc_macros::HashStable_Generic;

    // The subset of llvm::Attribute needed for arguments, packed into a bitfield.
    #[derive(Clone, Copy, Default, Hash, PartialEq, Eq, HashStable_Generic)]
    pub struct ArgAttribute(u8);
    bitflags::bitflags! {
        impl ArgAttribute: u8 {
            const NoAlias   = 1 << 1;
            const NoCapture = 1 << 2;
            const NonNull   = 1 << 3;
            const ReadOnly  = 1 << 4;
            const InReg     = 1 << 5;
            const NoUndef = 1 << 6;
        }
    }
    rustc_data_structures::external_bitflags_debug! { ArgAttribute }
}

/// Sometimes an ABI requires small integers to be extended to a full or partial register. This enum
/// defines if this extension should be zero-extension or sign-extension when necessary. When it is
/// not necessary to extend the argument, this enum is ignored.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum ArgExtension {
    None,
    Zext,
    Sext,
}

/// A compact representation of LLVM attributes (at least those relevant for this module)
/// that can be manipulated without interacting with LLVM's Attribute machinery.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct ArgAttributes {
    pub regular: ArgAttribute,
    pub arg_ext: ArgExtension,
    /// The minimum size of the pointee, guaranteed to be valid for the duration of the whole call
    /// (corresponding to LLVM's dereferenceable_or_null attributes, i.e., it is okay for this to be
    /// set on a null pointer, but all non-null pointers must be dereferenceable).
    pub pointee_size: Size,
    /// The minimum alignment of the pointee, if any.
    pub pointee_align: Option<Align>,
}

impl ArgAttributes {
    pub fn new() -> Self {
        ArgAttributes {
            regular: ArgAttribute::default(),
            arg_ext: ArgExtension::None,
            pointee_size: Size::ZERO,
            pointee_align: None,
        }
    }

    pub fn ext(&mut self, ext: ArgExtension) -> &mut Self {
        assert!(
            self.arg_ext == ArgExtension::None || self.arg_ext == ext,
            "cannot set {:?} when {:?} is already set",
            ext,
            self.arg_ext
        );
        self.arg_ext = ext;
        self
    }

    pub fn set(&mut self, attr: ArgAttribute) -> &mut Self {
        self.regular |= attr;
        self
    }

    pub fn contains(&self, attr: ArgAttribute) -> bool {
        self.regular.contains(attr)
    }

    /// Checks if these two `ArgAttributes` are equal enough to be considered "the same for all
    /// function call ABIs".
    pub fn eq_abi(&self, other: &Self) -> bool {
        // There's only one regular attribute that matters for the call ABI: InReg.
        // Everything else is things like noalias, dereferenceable, nonnull, ...
        // (This also applies to pointee_size, pointee_align.)
        if self.regular.contains(ArgAttribute::InReg) != other.regular.contains(ArgAttribute::InReg)
        {
            return false;
        }
        // We also compare the sign extension mode -- this could let the callee make assumptions
        // about bits that conceptually were not even passed.
        if self.arg_ext != other.arg_ext {
            return false;
        }
        true
    }
}

/// An argument passed entirely registers with the
/// same kind (e.g., HFA / HVA on PPC64 and AArch64).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct Uniform {
    pub unit: Reg,

    /// The total size of the argument, which can be:
    /// * equal to `unit.size` (one scalar/vector),
    /// * a multiple of `unit.size` (an array of scalar/vectors),
    /// * if `unit.kind` is `Integer`, the last element can be shorter, i.e., `{ i64, i64, i32 }`
    ///   for 64-bit integers with a total size of 20 bytes. When the argument is actually passed,
    ///   this size will be rounded up to the nearest multiple of `unit.size`.
    pub total: Size,

    /// Indicate that the argument is consecutive, in the sense that either all values need to be
    /// passed in register, or all on the stack. If they are passed on the stack, there should be
    /// no additional padding between elements.
    pub is_consecutive: bool,
}

impl From<Reg> for Uniform {
    fn from(unit: Reg) -> Uniform {
        Uniform { unit, total: unit.size, is_consecutive: false }
    }
}

impl Uniform {
    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        self.unit.align(cx)
    }

    /// Pass using one or more values of the given type, without requiring them to be consecutive.
    /// That is, some values may be passed in register and some on the stack.
    pub fn new(unit: Reg, total: Size) -> Self {
        Uniform { unit, total, is_consecutive: false }
    }

    /// Pass using one or more consecutive values of the given type. Either all values will be
    /// passed in registers, or all on the stack.
    pub fn consecutive(unit: Reg, total: Size) -> Self {
        Uniform { unit, total, is_consecutive: true }
    }
}

/// Describes the type used for `PassMode::Cast`.
///
/// Passing arguments in this mode works as follows: the registers in the `prefix` (the ones that
/// are `Some`) get laid out one after the other (using `repr(C)` layout rules). Then the
/// `rest.unit` register type gets repeated often enough to cover `rest.size`. This describes the
/// actual type used for the call; the Rust type of the argument is then transmuted to this ABI type
/// (and all data in the padding between the registers is dropped).
#[derive(Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct CastTarget {
    pub prefix: [Option<Reg>; 8],
    pub rest: Uniform,
    pub attrs: ArgAttributes,
}

impl From<Reg> for CastTarget {
    fn from(unit: Reg) -> CastTarget {
        CastTarget::from(Uniform::from(unit))
    }
}

impl From<Uniform> for CastTarget {
    fn from(uniform: Uniform) -> CastTarget {
        CastTarget {
            prefix: [None; 8],
            rest: uniform,
            attrs: ArgAttributes {
                regular: ArgAttribute::default(),
                arg_ext: ArgExtension::None,
                pointee_size: Size::ZERO,
                pointee_align: None,
            },
        }
    }
}

impl CastTarget {
    pub fn pair(a: Reg, b: Reg) -> CastTarget {
        CastTarget {
            prefix: [Some(a), None, None, None, None, None, None, None],
            rest: Uniform::from(b),
            attrs: ArgAttributes {
                regular: ArgAttribute::default(),
                arg_ext: ArgExtension::None,
                pointee_size: Size::ZERO,
                pointee_align: None,
            },
        }
    }

    /// When you only access the range containing valid data, you can use this unaligned size;
    /// otherwise, use the safer `size` method.
    pub fn unaligned_size<C: HasDataLayout>(&self, _cx: &C) -> Size {
        // Prefix arguments are passed in specific designated registers
        let prefix_size = self
            .prefix
            .iter()
            .filter_map(|x| x.map(|reg| reg.size))
            .fold(Size::ZERO, |acc, size| acc + size);
        // Remaining arguments are passed in chunks of the unit size
        let rest_size =
            self.rest.unit.size * self.rest.total.bytes().div_ceil(self.rest.unit.size.bytes());

        prefix_size + rest_size
    }

    pub fn size<C: HasDataLayout>(&self, cx: &C) -> Size {
        self.unaligned_size(cx).align_to(self.align(cx))
    }

    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        self.prefix
            .iter()
            .filter_map(|x| x.map(|reg| reg.align(cx)))
            .fold(cx.data_layout().aggregate_align.abi.max(self.rest.align(cx)), |acc, align| {
                acc.max(align)
            })
    }

    /// Checks if these two `CastTarget` are equal enough to be considered "the same for all
    /// function call ABIs".
    pub fn eq_abi(&self, other: &Self) -> bool {
        let CastTarget { prefix: prefix_l, rest: rest_l, attrs: attrs_l } = self;
        let CastTarget { prefix: prefix_r, rest: rest_r, attrs: attrs_r } = other;
        prefix_l == prefix_r && rest_l == rest_r && attrs_l.eq_abi(attrs_r)
    }
}

/// Information about how to pass an argument to,
/// or return a value from, a function, under some ABI.
#[derive(Clone, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct ArgAbi<'a, Ty> {
    pub layout: TyAndLayout<'a, Ty>,
    pub mode: PassMode,
}

// Needs to be a custom impl because of the bounds on the `TyAndLayout` debug impl.
impl<'a, Ty: fmt::Display> fmt::Debug for ArgAbi<'a, Ty> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let ArgAbi { layout, mode } = self;
        f.debug_struct("ArgAbi").field("layout", layout).field("mode", mode).finish()
    }
}

impl<'a, Ty> ArgAbi<'a, Ty> {
    /// This defines the "default ABI" for that type, that is then later adjusted in `fn_abi_adjust_for_abi`.
    pub fn new(
        cx: &impl HasDataLayout,
        layout: TyAndLayout<'a, Ty>,
        scalar_attrs: impl Fn(&TyAndLayout<'a, Ty>, Scalar, Size) -> ArgAttributes,
    ) -> Self {
        let mode = match layout.backend_repr {
            BackendRepr::Scalar(scalar) => {
                PassMode::Direct(scalar_attrs(&layout, scalar, Size::ZERO))
            }
            BackendRepr::ScalarPair(a, b) => PassMode::Pair(
                scalar_attrs(&layout, a, Size::ZERO),
                scalar_attrs(&layout, b, a.size(cx).align_to(b.align(cx).abi)),
            ),
            BackendRepr::SimdVector { .. } => PassMode::Direct(ArgAttributes::new()),
            BackendRepr::Memory { .. } => Self::indirect_pass_mode(&layout),
        };
        ArgAbi { layout, mode }
    }

    fn indirect_pass_mode(layout: &TyAndLayout<'a, Ty>) -> PassMode {
        let mut attrs = ArgAttributes::new();

        // For non-immediate arguments the callee gets its own copy of
        // the value on the stack, so there are no aliases. It's also
        // program-invisible so can't possibly capture
        attrs
            .set(ArgAttribute::NoAlias)
            .set(ArgAttribute::NoCapture)
            .set(ArgAttribute::NonNull)
            .set(ArgAttribute::NoUndef);
        attrs.pointee_size = layout.size;
        attrs.pointee_align = Some(layout.align.abi);

        let meta_attrs = layout.is_unsized().then_some(ArgAttributes::new());

        PassMode::Indirect { attrs, meta_attrs, on_stack: false }
    }

    /// Pass this argument directly instead. Should NOT be used!
    /// Only exists because of past ABI mistakes that will take time to fix
    /// (see <https://github.com/rust-lang/rust/issues/115666>).
    #[track_caller]
    pub fn make_direct_deprecated(&mut self) {
        match self.mode {
            PassMode::Indirect { .. } => {
                self.mode = PassMode::Direct(ArgAttributes::new());
            }
            PassMode::Ignore | PassMode::Direct(_) | PassMode::Pair(_, _) => {} // already direct
            _ => panic!("Tried to make {:?} direct", self.mode),
        }
    }

    /// Pass this argument indirectly, by passing a (thin or wide) pointer to the argument instead.
    /// This is valid for both sized and unsized arguments.
    #[track_caller]
    pub fn make_indirect(&mut self) {
        match self.mode {
            PassMode::Direct(_) | PassMode::Pair(_, _) => {
                self.mode = Self::indirect_pass_mode(&self.layout);
            }
            PassMode::Indirect { attrs: _, meta_attrs: _, on_stack: false } => {
                // already indirect
            }
            _ => panic!("Tried to make {:?} indirect", self.mode),
        }
    }

    /// Same as `make_indirect`, but for arguments that are ignored. Only needed for ABIs that pass
    /// ZSTs indirectly.
    #[track_caller]
    pub fn make_indirect_from_ignore(&mut self) {
        match self.mode {
            PassMode::Ignore => {
                self.mode = Self::indirect_pass_mode(&self.layout);
            }
            PassMode::Indirect { attrs: _, meta_attrs: _, on_stack: false } => {
                // already indirect
            }
            _ => panic!("Tried to make {:?} indirect (expected `PassMode::Ignore`)", self.mode),
        }
    }

    /// Pass this argument indirectly, by placing it at a fixed stack offset.
    /// This corresponds to the `byval` LLVM argument attribute.
    /// This is only valid for sized arguments.
    ///
    /// `byval_align` specifies the alignment of the `byval` stack slot, which does not need to
    /// correspond to the type's alignment. This will be `Some` if the target's ABI specifies that
    /// stack slots used for arguments passed by-value have specific alignment requirements which
    /// differ from the alignment used in other situations.
    ///
    /// If `None`, the type's alignment is used.
    ///
    /// If the resulting alignment differs from the type's alignment,
    /// the argument will be copied to an alloca with sufficient alignment,
    /// either in the caller (if the type's alignment is lower than the byval alignment)
    /// or in the callee (if the type's alignment is higher than the byval alignment),
    /// to ensure that Rust code never sees an underaligned pointer.
    pub fn pass_by_stack_offset(&mut self, byval_align: Option<Align>) {
        assert!(!self.layout.is_unsized(), "used byval ABI for unsized layout");
        self.make_indirect();
        match self.mode {
            PassMode::Indirect { ref mut attrs, meta_attrs: _, ref mut on_stack } => {
                *on_stack = true;

                // Some platforms, like 32-bit x86, change the alignment of the type when passing
                // `byval`. Account for that.
                if let Some(byval_align) = byval_align {
                    // On all targets with byval align this is currently true, so let's assert it.
                    debug_assert!(byval_align >= Align::from_bytes(4).unwrap());
                    attrs.pointee_align = Some(byval_align);
                }
            }
            _ => unreachable!(),
        }
    }

    pub fn extend_integer_width_to(&mut self, bits: u64) {
        // Only integers have signedness
        if let BackendRepr::Scalar(scalar) = self.layout.backend_repr {
            if let Primitive::Int(i, signed) = scalar.primitive() {
                if i.size().bits() < bits {
                    if let PassMode::Direct(ref mut attrs) = self.mode {
                        if signed {
                            attrs.ext(ArgExtension::Sext)
                        } else {
                            attrs.ext(ArgExtension::Zext)
                        };
                    }
                }
            }
        }
    }

    pub fn cast_to<T: Into<CastTarget>>(&mut self, target: T) {
        self.mode = PassMode::Cast { cast: Box::new(target.into()), pad_i32: false };
    }

    pub fn cast_to_and_pad_i32<T: Into<CastTarget>>(&mut self, target: T, pad_i32: bool) {
        self.mode = PassMode::Cast { cast: Box::new(target.into()), pad_i32 };
    }

    pub fn is_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { .. })
    }

    pub fn is_sized_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { attrs: _, meta_attrs: None, on_stack: _ })
    }

    pub fn is_unsized_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { attrs: _, meta_attrs: Some(_), on_stack: _ })
    }

    pub fn is_ignore(&self) -> bool {
        matches!(self.mode, PassMode::Ignore)
    }

    /// Checks if these two `ArgAbi` are equal enough to be considered "the same for all
    /// function call ABIs".
    pub fn eq_abi(&self, other: &Self) -> bool
    where
        Ty: PartialEq,
    {
        // Ideally we'd just compare the `mode`, but that is not enough -- for some modes LLVM will look
        // at the type.
        self.layout.eq_abi(&other.layout) && self.mode.eq_abi(&other.mode) && {
            // `fn_arg_sanity_check` accepts `PassMode::Direct` for some aggregates.
            // That elevates any type difference to an ABI difference since we just use the
            // full Rust type as the LLVM argument/return type.
            if matches!(self.mode, PassMode::Direct(..))
                && matches!(self.layout.backend_repr, BackendRepr::Memory { .. })
            {
                // For aggregates in `Direct` mode to be compatible, the types need to be equal.
                self.layout.ty == other.layout.ty
            } else {
                true
            }
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum RiscvInterruptKind {
    Machine,
    Supervisor,
}

impl RiscvInterruptKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Machine => "machine",
            Self::Supervisor => "supervisor",
        }
    }
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// The signature represented by this type may not match the MIR function signature.
/// Certain attributes, like `#[track_caller]` can introduce additional arguments, which are present in [`FnAbi`], but not in `FnSig`.
/// While this difference is rarely relevant, it should still be kept in mind.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
#[derive(Clone, PartialEq, Eq, Hash, HashStable_Generic)]
pub struct FnAbi<'a, Ty> {
    /// The type, layout, and information about how each argument is passed.
    pub args: Box<[ArgAbi<'a, Ty>]>,

    /// The layout, type, and the way a value is returned from this function.
    pub ret: ArgAbi<'a, Ty>,

    /// Marks this function as variadic (accepting a variable number of arguments).
    pub c_variadic: bool,

    /// The count of non-variadic arguments.
    ///
    /// Should only be different from args.len() when c_variadic is true.
    /// This can be used to know whether an argument is variadic or not.
    pub fixed_count: u32,
    /// The calling convention of this function.
    pub conv: CanonAbi,
    /// Indicates if an unwind may happen across a call to this function.
    pub can_unwind: bool,
}

// Needs to be a custom impl because of the bounds on the `TyAndLayout` debug impl.
impl<'a, Ty: fmt::Display> fmt::Debug for FnAbi<'a, Ty> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let FnAbi { args, ret, c_variadic, fixed_count, conv, can_unwind } = self;
        f.debug_struct("FnAbi")
            .field("args", args)
            .field("ret", ret)
            .field("c_variadic", c_variadic)
            .field("fixed_count", fixed_count)
            .field("conv", conv)
            .field("can_unwind", can_unwind)
            .finish()
    }
}

impl<'a, Ty> FnAbi<'a, Ty> {
    pub fn adjust_for_foreign_abi<C>(&mut self, cx: &C, abi: ExternAbi)
    where
        Ty: TyAbiInterface<'a, C> + Copy,
        C: HasDataLayout + HasTargetSpec + HasWasmCAbiOpt + HasX86AbiOpt,
    {
        if abi == ExternAbi::X86Interrupt {
            if let Some(arg) = self.args.first_mut() {
                arg.pass_by_stack_offset(None);
            }
            return;
        }

        let spec = cx.target_spec();
        match &spec.arch[..] {
            "x86" => {
                let (flavor, regparm) = match abi {
                    ExternAbi::Fastcall { .. } | ExternAbi::Vectorcall { .. } => {
                        (x86::Flavor::FastcallOrVectorcall, None)
                    }
                    ExternAbi::C { .. } | ExternAbi::Cdecl { .. } | ExternAbi::Stdcall { .. } => {
                        (x86::Flavor::General, cx.x86_abi_opt().regparm)
                    }
                    _ => (x86::Flavor::General, None),
                };
                let reg_struct_return = cx.x86_abi_opt().reg_struct_return;
                let opts = x86::X86Options { flavor, regparm, reg_struct_return };
                if spec.is_like_msvc {
                    x86_win32::compute_abi_info(cx, self, opts);
                } else {
                    x86::compute_abi_info(cx, self, opts);
                }
            }
            "x86_64" => match abi {
                ExternAbi::SysV64 { .. } => x86_64::compute_abi_info(cx, self),
                ExternAbi::Win64 { .. } | ExternAbi::Vectorcall { .. } => {
                    x86_win64::compute_abi_info(cx, self)
                }
                _ => {
                    if cx.target_spec().is_like_windows {
                        x86_win64::compute_abi_info(cx, self)
                    } else {
                        x86_64::compute_abi_info(cx, self)
                    }
                }
            },
            "aarch64" | "arm64ec" => {
                let kind = if cx.target_spec().is_like_darwin {
                    aarch64::AbiKind::DarwinPCS
                } else if cx.target_spec().is_like_windows {
                    aarch64::AbiKind::Win64
                } else {
                    aarch64::AbiKind::AAPCS
                };
                aarch64::compute_abi_info(cx, self, kind)
            }
            "amdgpu" => amdgpu::compute_abi_info(cx, self),
            "arm" => arm::compute_abi_info(cx, self),
            "avr" => avr::compute_abi_info(self),
            "loongarch64" => loongarch::compute_abi_info(cx, self),
            "m68k" => m68k::compute_abi_info(self),
            "csky" => csky::compute_abi_info(self),
            "mips" | "mips32r6" => mips::compute_abi_info(cx, self),
            "mips64" | "mips64r6" => mips64::compute_abi_info(cx, self),
            "powerpc" => powerpc::compute_abi_info(cx, self),
            "powerpc64" => powerpc64::compute_abi_info(cx, self),
            "s390x" => s390x::compute_abi_info(cx, self),
            "msp430" => msp430::compute_abi_info(self),
            "sparc" => sparc::compute_abi_info(cx, self),
            "sparc64" => sparc64::compute_abi_info(cx, self),
            "nvptx64" => {
                if abi == ExternAbi::PtxKernel || abi == ExternAbi::GpuKernel {
                    nvptx64::compute_ptx_kernel_abi_info(cx, self)
                } else {
                    nvptx64::compute_abi_info(self)
                }
            }
            "hexagon" => hexagon::compute_abi_info(self),
            "xtensa" => xtensa::compute_abi_info(cx, self),
            "riscv32" | "riscv64" => riscv::compute_abi_info(cx, self),
            "wasm32" => {
                if spec.os == "unknown" && matches!(cx.wasm_c_abi_opt(), WasmCAbi::Legacy { .. }) {
                    wasm::compute_wasm_abi_info(self)
                } else {
                    wasm::compute_c_abi_info(cx, self)
                }
            }
            "wasm64" => wasm::compute_c_abi_info(cx, self),
            "bpf" => bpf::compute_abi_info(self),
            arch => panic!("no lowering implemented for {arch}"),
        }
    }

    pub fn adjust_for_rust_abi<C>(&mut self, cx: &C)
    where
        Ty: TyAbiInterface<'a, C> + Copy,
        C: HasDataLayout + HasTargetSpec,
    {
        let spec = cx.target_spec();
        match &*spec.arch {
            "x86" => x86::compute_rust_abi_info(cx, self),
            "riscv32" | "riscv64" => riscv::compute_rust_abi_info(cx, self),
            "loongarch64" => loongarch::compute_rust_abi_info(cx, self),
            "aarch64" => aarch64::compute_rust_abi_info(cx, self),
            _ => {}
        };

        // Decides whether we can pass the given SIMD argument via `PassMode::Direct`.
        // May only return `true` if the target will always pass those arguments the same way,
        // no matter what the user does with `-Ctarget-feature`! In other words, whatever
        // target features are required to pass a SIMD value in registers must be listed in
        // the `abi_required_features` for the current target and ABI.
        let can_pass_simd_directly = |arg: &ArgAbi<'_, Ty>| match &*spec.arch {
            // On x86, if we have SSE2 (which we have by default for x86_64), we can always pass up
            // to 128-bit-sized vectors.
            "x86" if spec.rustc_abi == Some(RustcAbi::X86Sse2) => arg.layout.size.bits() <= 128,
            "x86_64" if spec.rustc_abi != Some(RustcAbi::X86Softfloat) => {
                // FIXME once https://github.com/bytecodealliance/wasmtime/issues/10254 is fixed
                // accept vectors up to 128bit rather than vectors of exactly 128bit.
                arg.layout.size.bits() == 128
            }
            // So far, we haven't implemented this logic for any other target.
            _ => false,
        };

        for (arg_idx, arg) in self
            .args
            .iter_mut()
            .enumerate()
            .map(|(idx, arg)| (Some(idx), arg))
            .chain(iter::once((None, &mut self.ret)))
        {
            // If the logic above already picked a specific type to cast the argument to, leave that
            // in place.
            if matches!(arg.mode, PassMode::Ignore | PassMode::Cast { .. }) {
                continue;
            }

            if arg_idx.is_none()
                && arg.layout.size > Primitive::Pointer(AddressSpace::DATA).size(cx) * 2
                && !matches!(arg.layout.backend_repr, BackendRepr::SimdVector { .. })
            {
                // Return values larger than 2 registers using a return area
                // pointer. LLVM and Cranelift disagree about how to return
                // values that don't fit in the registers designated for return
                // values. LLVM will force the entire return value to be passed
                // by return area pointer, while Cranelift will look at each IR level
                // return value independently and decide to pass it in a
                // register or not, which would result in the return value
                // being passed partially in registers and partially through a
                // return area pointer. For large IR-level values such as `i128`,
                // cranelift will even split up the value into smaller chunks.
                //
                // While Cranelift may need to be fixed as the LLVM behavior is
                // generally more correct with respect to the surface language,
                // forcing this behavior in rustc itself makes it easier for
                // other backends to conform to the Rust ABI and for the C ABI
                // rustc already handles this behavior anyway.
                //
                // In addition LLVM's decision to pass the return value in
                // registers or using a return area pointer depends on how
                // exactly the return type is lowered to an LLVM IR type. For
                // example `Option<u128>` can be lowered as `{ i128, i128 }`
                // in which case the x86_64 backend would use a return area
                // pointer, or it could be passed as `{ i32, i128 }` in which
                // case the x86_64 backend would pass it in registers by taking
                // advantage of an LLVM ABI extension that allows using 3
                // registers for the x86_64 sysv call conv rather than the
                // officially specified 2 registers.
                //
                // FIXME: Technically we should look at the amount of available
                // return registers rather than guessing that there are 2
                // registers for return values. In practice only a couple of
                // architectures have less than 2 return registers. None of
                // which supported by Cranelift.
                //
                // NOTE: This adjustment is only necessary for the Rust ABI as
                // for other ABI's the calling convention implementations in
                // rustc_target already ensure any return value which doesn't
                // fit in the available amount of return registers is passed in
                // the right way for the current target.
                //
                // The adjustment is not necessary nor desired for types with a vector
                // representation; those are handled below.
                arg.make_indirect();
                continue;
            }

            match arg.layout.backend_repr {
                BackendRepr::Memory { .. } => {
                    // Compute `Aggregate` ABI.

                    let is_indirect_not_on_stack =
                        matches!(arg.mode, PassMode::Indirect { on_stack: false, .. });
                    assert!(is_indirect_not_on_stack);

                    let size = arg.layout.size;
                    if arg.layout.is_sized()
                        && size <= Primitive::Pointer(AddressSpace::DATA).size(cx)
                    {
                        // We want to pass small aggregates as immediates, but using
                        // an LLVM aggregate type for this leads to bad optimizations,
                        // so we pick an appropriately sized integer type instead.
                        arg.cast_to(Reg { kind: RegKind::Integer, size });
                    }
                }

                BackendRepr::SimdVector { .. } => {
                    // This is a fun case! The gist of what this is doing is
                    // that we want callers and callees to always agree on the
                    // ABI of how they pass SIMD arguments. If we were to *not*
                    // make these arguments indirect then they'd be immediates
                    // in LLVM, which means that they'd used whatever the
                    // appropriate ABI is for the callee and the caller. That
                    // means, for example, if the caller doesn't have AVX
                    // enabled but the callee does, then passing an AVX argument
                    // across this boundary would cause corrupt data to show up.
                    //
                    // This problem is fixed by unconditionally passing SIMD
                    // arguments through memory between callers and callees
                    // which should get them all to agree on ABI regardless of
                    // target feature sets. Some more information about this
                    // issue can be found in #44367.
                    //
                    // Note that the intrinsic ABI is exempt here as those are not
                    // real functions anyway, and the backend expects very specific types.
                    if spec.simd_types_indirect && !can_pass_simd_directly(arg) {
                        arg.make_indirect();
                    }
                }

                _ => {}
            }
        }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(target_pointer_width = "64")]
mod size_asserts {
    use rustc_data_structures::static_assert_size;

    use super::*;
    // tidy-alphabetical-start
    static_assert_size!(ArgAbi<'_, usize>, 56);
    static_assert_size!(FnAbi<'_, usize>, 80);
    // tidy-alphabetical-end
}

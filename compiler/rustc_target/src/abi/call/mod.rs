use crate::abi::{self, Abi, Align, FieldsShape, Size};
use crate::abi::{HasDataLayout, TyAbiInterface, TyAndLayout};
use crate::spec::{self, HasTargetSpec};
use rustc_span::Symbol;
use std::fmt;
use std::str::FromStr;

mod aarch64;
mod amdgpu;
mod arm;
mod avr;
mod bpf;
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
mod x86_win64;

#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum PassMode {
    /// Ignore the argument.
    ///
    /// The argument is either uninhabited or a ZST.
    Ignore,
    /// Pass the argument directly.
    ///
    /// The argument has a layout abi of `Scalar`, `Vector` or in rare cases `Aggregate`.
    Direct(ArgAttributes),
    /// Pass a pair's elements directly in two arguments.
    ///
    /// The argument has a layout abi of `ScalarPair`.
    Pair(ArgAttributes, ArgAttributes),
    /// Pass the argument after casting it, to either a single uniform or a
    /// pair of registers. The bool indicates if a `Reg::i32()` dummy argument
    /// is emitted before the real argument.
    Cast(Box<CastTarget>, bool),
    /// Pass the argument indirectly via a hidden pointer.
    /// The `extra_attrs` value, if any, is for the extra data (vtable or length)
    /// which indicates that it refers to an unsized rvalue.
    /// `on_stack` defines that the value should be passed at a fixed
    /// stack offset in accordance to the ABI rather than passed using a
    /// pointer. This corresponds to the `byval` LLVM argument attribute.
    Indirect { attrs: ArgAttributes, extra_attrs: Option<ArgAttributes>, on_stack: bool },
}

// Hack to disable non_upper_case_globals only for the bitflags! and not for the rest
// of this module
pub use attr_impl::ArgAttribute;

#[allow(non_upper_case_globals)]
#[allow(unused)]
mod attr_impl {
    // The subset of llvm::Attribute needed for arguments, packed into a bitfield.
    bitflags::bitflags! {
        #[derive(Default, HashStable_Generic)]
        pub struct ArgAttribute: u8 {
            const NoAlias   = 1 << 1;
            const NoCapture = 1 << 2;
            const NonNull   = 1 << 3;
            const ReadOnly  = 1 << 4;
            const InReg     = 1 << 5;
            const NoUndef = 1 << 6;
        }
    }
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
    /// (corresponding to LLVM's dereferenceable and dereferenceable_or_null attributes).
    pub pointee_size: Size,
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
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum RegKind {
    Integer,
    Float,
    Vector,
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct Reg {
    pub kind: RegKind,
    pub size: Size,
}

macro_rules! reg_ctor {
    ($name:ident, $kind:ident, $bits:expr) => {
        pub fn $name() -> Reg {
            Reg { kind: RegKind::$kind, size: Size::from_bits($bits) }
        }
    };
}

impl Reg {
    reg_ctor!(i8, Integer, 8);
    reg_ctor!(i16, Integer, 16);
    reg_ctor!(i32, Integer, 32);
    reg_ctor!(i64, Integer, 64);
    reg_ctor!(i128, Integer, 128);

    reg_ctor!(f32, Float, 32);
    reg_ctor!(f64, Float, 64);
}

impl Reg {
    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        let dl = cx.data_layout();
        match self.kind {
            RegKind::Integer => match self.size.bits() {
                1 => dl.i1_align.abi,
                2..=8 => dl.i8_align.abi,
                9..=16 => dl.i16_align.abi,
                17..=32 => dl.i32_align.abi,
                33..=64 => dl.i64_align.abi,
                65..=128 => dl.i128_align.abi,
                _ => panic!("unsupported integer: {self:?}"),
            },
            RegKind::Float => match self.size.bits() {
                32 => dl.f32_align.abi,
                64 => dl.f64_align.abi,
                _ => panic!("unsupported float: {self:?}"),
            },
            RegKind::Vector => dl.vector_align(self.size).abi,
        }
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
    /// * if `unit.kind` is `Integer`, the last element
    ///   can be shorter, i.e., `{ i64, i64, i32 }` for
    ///   64-bit integers with a total size of 20 bytes.
    pub total: Size,
}

impl From<Reg> for Uniform {
    fn from(unit: Reg) -> Uniform {
        Uniform { unit, total: unit.size }
    }
}

impl Uniform {
    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        self.unit.align(cx)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
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

    pub fn size<C: HasDataLayout>(&self, _cx: &C) -> Size {
        let mut size = self.rest.total;
        for i in 0..self.prefix.iter().count() {
            match self.prefix[i] {
                Some(v) => size += v.size,
                None => {}
            }
        }
        return size;
    }

    pub fn align<C: HasDataLayout>(&self, cx: &C) -> Align {
        self.prefix
            .iter()
            .filter_map(|x| x.map(|reg| reg.align(cx)))
            .fold(cx.data_layout().aggregate_align.abi.max(self.rest.align(cx)), |acc, align| {
                acc.max(align)
            })
    }
}

/// Return value from the `homogeneous_aggregate` test function.
#[derive(Copy, Clone, Debug)]
pub enum HomogeneousAggregate {
    /// Yes, all the "leaf fields" of this struct are passed in the
    /// same way (specified in the `Reg` value).
    Homogeneous(Reg),

    /// There are no leaf fields at all.
    NoData,
}

/// Error from the `homogeneous_aggregate` test function, indicating
/// there are distinct leaf fields passed in different ways,
/// or this is uninhabited.
#[derive(Copy, Clone, Debug)]
pub struct Heterogeneous;

impl HomogeneousAggregate {
    /// If this is a homogeneous aggregate, returns the homogeneous
    /// unit, else `None`.
    pub fn unit(self) -> Option<Reg> {
        match self {
            HomogeneousAggregate::Homogeneous(reg) => Some(reg),
            HomogeneousAggregate::NoData => None,
        }
    }

    /// Try to combine two `HomogeneousAggregate`s, e.g. from two fields in
    /// the same `struct`. Only succeeds if only one of them has any data,
    /// or both units are identical.
    fn merge(self, other: HomogeneousAggregate) -> Result<HomogeneousAggregate, Heterogeneous> {
        match (self, other) {
            (x, HomogeneousAggregate::NoData) | (HomogeneousAggregate::NoData, x) => Ok(x),

            (HomogeneousAggregate::Homogeneous(a), HomogeneousAggregate::Homogeneous(b)) => {
                if a != b {
                    return Err(Heterogeneous);
                }
                Ok(self)
            }
        }
    }
}

impl<'a, Ty> TyAndLayout<'a, Ty> {
    fn is_aggregate(&self) -> bool {
        match self.abi {
            Abi::Uninhabited | Abi::Scalar(_) | Abi::Vector { .. } => false,
            Abi::ScalarPair(..) | Abi::Aggregate { .. } => true,
        }
    }

    /// Returns `Homogeneous` if this layout is an aggregate containing fields of
    /// only a single type (e.g., `(u32, u32)`). Such aggregates are often
    /// special-cased in ABIs.
    ///
    /// Note: We generally ignore fields of zero-sized type when computing
    /// this value (see #56877).
    ///
    /// This is public so that it can be used in unit tests, but
    /// should generally only be relevant to the ABI details of
    /// specific targets.
    pub fn homogeneous_aggregate<C>(&self, cx: &C) -> Result<HomogeneousAggregate, Heterogeneous>
    where
        Ty: TyAbiInterface<'a, C> + Copy,
    {
        match self.abi {
            Abi::Uninhabited => Err(Heterogeneous),

            // The primitive for this algorithm.
            Abi::Scalar(scalar) => {
                let kind = match scalar.primitive() {
                    abi::Int(..) | abi::Pointer(_) => RegKind::Integer,
                    abi::F32 | abi::F64 => RegKind::Float,
                };
                Ok(HomogeneousAggregate::Homogeneous(Reg { kind, size: self.size }))
            }

            Abi::Vector { .. } => {
                assert!(!self.is_zst());
                Ok(HomogeneousAggregate::Homogeneous(Reg {
                    kind: RegKind::Vector,
                    size: self.size,
                }))
            }

            Abi::ScalarPair(..) | Abi::Aggregate { .. } => {
                // Helper for computing `homogeneous_aggregate`, allowing a custom
                // starting offset (used below for handling variants).
                let from_fields_at =
                    |layout: Self,
                     start: Size|
                     -> Result<(HomogeneousAggregate, Size), Heterogeneous> {
                        let is_union = match layout.fields {
                            FieldsShape::Primitive => {
                                unreachable!("aggregates can't have `FieldsShape::Primitive`")
                            }
                            FieldsShape::Array { count, .. } => {
                                assert_eq!(start, Size::ZERO);

                                let result = if count > 0 {
                                    layout.field(cx, 0).homogeneous_aggregate(cx)?
                                } else {
                                    HomogeneousAggregate::NoData
                                };
                                return Ok((result, layout.size));
                            }
                            FieldsShape::Union(_) => true,
                            FieldsShape::Arbitrary { .. } => false,
                        };

                        let mut result = HomogeneousAggregate::NoData;
                        let mut total = start;

                        for i in 0..layout.fields.count() {
                            if !is_union && total != layout.fields.offset(i) {
                                return Err(Heterogeneous);
                            }

                            let field = layout.field(cx, i);

                            result = result.merge(field.homogeneous_aggregate(cx)?)?;

                            // Keep track of the offset (without padding).
                            let size = field.size;
                            if is_union {
                                total = total.max(size);
                            } else {
                                total += size;
                            }
                        }

                        Ok((result, total))
                    };

                let (mut result, mut total) = from_fields_at(*self, Size::ZERO)?;

                match &self.variants {
                    abi::Variants::Single { .. } => {}
                    abi::Variants::Multiple { variants, .. } => {
                        // Treat enum variants like union members.
                        // HACK(eddyb) pretend the `enum` field (discriminant)
                        // is at the start of every variant (otherwise the gap
                        // at the start of all variants would disqualify them).
                        //
                        // NB: for all tagged `enum`s (which include all non-C-like
                        // `enum`s with defined FFI representation), this will
                        // match the homogeneous computation on the equivalent
                        // `struct { tag; union { variant1; ... } }` and/or
                        // `union { struct { tag; variant1; } ... }`
                        // (the offsets of variant fields should be identical
                        // between the two for either to be a homogeneous aggregate).
                        let variant_start = total;
                        for variant_idx in variants.indices() {
                            let (variant_result, variant_total) =
                                from_fields_at(self.for_variant(cx, variant_idx), variant_start)?;

                            result = result.merge(variant_result)?;
                            total = total.max(variant_total);
                        }
                    }
                }

                // There needs to be no padding.
                if total != self.size {
                    Err(Heterogeneous)
                } else {
                    match result {
                        HomogeneousAggregate::Homogeneous(_) => {
                            assert_ne!(total, Size::ZERO);
                        }
                        HomogeneousAggregate::NoData => {
                            assert_eq!(total, Size::ZERO);
                        }
                    }
                    Ok(result)
                }
            }
        }
    }
}

/// Information about how to pass an argument to,
/// or return a value from, a function, under some ABI.
#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct ArgAbi<'a, Ty> {
    pub layout: TyAndLayout<'a, Ty>,
    pub mode: PassMode,
}

impl<'a, Ty> ArgAbi<'a, Ty> {
    pub fn new(
        cx: &impl HasDataLayout,
        layout: TyAndLayout<'a, Ty>,
        scalar_attrs: impl Fn(&TyAndLayout<'a, Ty>, abi::Scalar, Size) -> ArgAttributes,
    ) -> Self {
        let mode = match layout.abi {
            Abi::Uninhabited => PassMode::Ignore,
            Abi::Scalar(scalar) => PassMode::Direct(scalar_attrs(&layout, scalar, Size::ZERO)),
            Abi::ScalarPair(a, b) => PassMode::Pair(
                scalar_attrs(&layout, a, Size::ZERO),
                scalar_attrs(&layout, b, a.size(cx).align_to(b.align(cx).abi)),
            ),
            Abi::Vector { .. } => PassMode::Direct(ArgAttributes::new()),
            Abi::Aggregate { .. } => PassMode::Direct(ArgAttributes::new()),
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
        // FIXME(eddyb) We should be doing this, but at least on
        // i686-pc-windows-msvc, it results in wrong stack offsets.
        // attrs.pointee_align = Some(layout.align.abi);

        let extra_attrs = layout.is_unsized().then_some(ArgAttributes::new());

        PassMode::Indirect { attrs, extra_attrs, on_stack: false }
    }

    pub fn make_indirect(&mut self) {
        match self.mode {
            PassMode::Direct(_) | PassMode::Pair(_, _) => {}
            PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: false } => return,
            _ => panic!("Tried to make {:?} indirect", self.mode),
        }

        self.mode = Self::indirect_pass_mode(&self.layout);
    }

    pub fn make_indirect_byval(&mut self) {
        self.make_indirect();
        match self.mode {
            PassMode::Indirect { attrs: _, extra_attrs: _, ref mut on_stack } => {
                *on_stack = true;
            }
            _ => unreachable!(),
        }
    }

    pub fn extend_integer_width_to(&mut self, bits: u64) {
        // Only integers have signedness
        if let Abi::Scalar(scalar) = self.layout.abi {
            if let abi::Int(i, signed) = scalar.primitive() {
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
        self.mode = PassMode::Cast(Box::new(target.into()), false);
    }

    pub fn cast_to_and_pad_i32<T: Into<CastTarget>>(&mut self, target: T, pad_i32: bool) {
        self.mode = PassMode::Cast(Box::new(target.into()), pad_i32);
    }

    pub fn is_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { .. })
    }

    pub fn is_sized_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { attrs: _, extra_attrs: None, on_stack: _ })
    }

    pub fn is_unsized_indirect(&self) -> bool {
        matches!(self.mode, PassMode::Indirect { attrs: _, extra_attrs: Some(_), on_stack: _ })
    }

    pub fn is_ignore(&self) -> bool {
        matches!(self.mode, PassMode::Ignore)
    }
}

#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub enum Conv {
    // General language calling conventions, for which every target
    // should have its own backend (e.g. LLVM) support.
    C,
    Rust,

    /// For things unlikely to be called, where smaller caller codegen is
    /// preferred over raw speed.
    /// Stronger than just `#[cold]` because `fn` pointers might be incompatible.
    RustCold,

    // Target-specific calling conventions.
    ArmAapcs,
    CCmseNonSecureCall,

    Msp430Intr,

    PtxKernel,

    X86Fastcall,
    X86Intr,
    X86Stdcall,
    X86ThisCall,
    X86VectorCall,

    X86_64SysV,
    X86_64Win64,

    AmdGpuKernel,
    AvrInterrupt,
    AvrNonBlockingInterrupt,
}

/// Metadata describing how the arguments to a native function
/// should be passed in order to respect the native ABI.
///
/// I will do my best to describe this structure, but these
/// comments are reverse-engineered and may be inaccurate. -NDM
#[derive(PartialEq, Eq, Hash, Debug, HashStable_Generic)]
pub struct FnAbi<'a, Ty> {
    /// The LLVM types of each argument.
    pub args: Box<[ArgAbi<'a, Ty>]>,

    /// LLVM return type.
    pub ret: ArgAbi<'a, Ty>,

    pub c_variadic: bool,

    /// The count of non-variadic arguments.
    ///
    /// Should only be different from args.len() when c_variadic is true.
    /// This can be used to know whether an argument is variadic or not.
    pub fixed_count: u32,

    pub conv: Conv,

    pub can_unwind: bool,
}

/// Error produced by attempting to adjust a `FnAbi`, for a "foreign" ABI.
#[derive(Copy, Clone, Debug, HashStable_Generic)]
pub enum AdjustForForeignAbiError {
    /// Target architecture doesn't support "foreign" (i.e. non-Rust) ABIs.
    Unsupported { arch: Symbol, abi: spec::abi::Abi },
}

impl fmt::Display for AdjustForForeignAbiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported { arch, abi } => {
                write!(f, "target architecture {arch:?} does not support `extern {abi}` ABI")
            }
        }
    }
}

impl<'a, Ty> FnAbi<'a, Ty> {
    pub fn adjust_for_foreign_abi<C>(
        &mut self,
        cx: &C,
        abi: spec::abi::Abi,
    ) -> Result<(), AdjustForForeignAbiError>
    where
        Ty: TyAbiInterface<'a, C> + Copy,
        C: HasDataLayout + HasTargetSpec,
    {
        if abi == spec::abi::Abi::X86Interrupt {
            if let Some(arg) = self.args.first_mut() {
                arg.make_indirect_byval();
            }
            return Ok(());
        }

        match &cx.target_spec().arch[..] {
            "x86" => {
                let flavor = if let spec::abi::Abi::Fastcall { .. }
                | spec::abi::Abi::Vectorcall { .. } = abi
                {
                    x86::Flavor::FastcallOrVectorcall
                } else {
                    x86::Flavor::General
                };
                x86::compute_abi_info(cx, self, flavor);
            }
            "x86_64" => match abi {
                spec::abi::Abi::SysV64 { .. } => x86_64::compute_abi_info(cx, self),
                spec::abi::Abi::Win64 { .. } => x86_win64::compute_abi_info(self),
                _ => {
                    if cx.target_spec().is_like_windows {
                        x86_win64::compute_abi_info(self)
                    } else {
                        x86_64::compute_abi_info(cx, self)
                    }
                }
            },
            "aarch64" => {
                let param_policy = if cx.target_spec().is_like_osx {
                    aarch64::ParamExtension::ExtendTo32Bits
                } else {
                    aarch64::ParamExtension::NoExtension
                };
                aarch64::compute_abi_info(cx, self, param_policy)
            }
            "amdgpu" => amdgpu::compute_abi_info(cx, self),
            "arm" => arm::compute_abi_info(cx, self),
            "avr" => avr::compute_abi_info(self),
            "loongarch64" => loongarch::compute_abi_info(cx, self),
            "m68k" => m68k::compute_abi_info(self),
            "mips" => mips::compute_abi_info(cx, self),
            "mips64" => mips64::compute_abi_info(cx, self),
            "powerpc" => powerpc::compute_abi_info(self),
            "powerpc64" => powerpc64::compute_abi_info(cx, self),
            "s390x" => s390x::compute_abi_info(cx, self),
            "msp430" => msp430::compute_abi_info(self),
            "sparc" => sparc::compute_abi_info(cx, self),
            "sparc64" => sparc64::compute_abi_info(cx, self),
            "nvptx64" => {
                if cx.target_spec().adjust_abi(abi) == spec::abi::Abi::PtxKernel {
                    nvptx64::compute_ptx_kernel_abi_info(cx, self)
                } else {
                    nvptx64::compute_abi_info(self)
                }
            }
            "hexagon" => hexagon::compute_abi_info(self),
            "riscv32" | "riscv64" => riscv::compute_abi_info(cx, self),
            "wasm32" | "wasm64" => {
                if cx.target_spec().adjust_abi(abi) == spec::abi::Abi::Wasm {
                    wasm::compute_wasm_abi_info(self)
                } else {
                    wasm::compute_c_abi_info(cx, self)
                }
            }
            "asmjs" => wasm::compute_c_abi_info(cx, self),
            "bpf" => bpf::compute_abi_info(self),
            arch => {
                return Err(AdjustForForeignAbiError::Unsupported {
                    arch: Symbol::intern(arch),
                    abi,
                });
            }
        }

        Ok(())
    }
}

impl FromStr for Conv {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "C" => Ok(Conv::C),
            "Rust" => Ok(Conv::Rust),
            "RustCold" => Ok(Conv::Rust),
            "ArmAapcs" => Ok(Conv::ArmAapcs),
            "CCmseNonSecureCall" => Ok(Conv::CCmseNonSecureCall),
            "Msp430Intr" => Ok(Conv::Msp430Intr),
            "PtxKernel" => Ok(Conv::PtxKernel),
            "X86Fastcall" => Ok(Conv::X86Fastcall),
            "X86Intr" => Ok(Conv::X86Intr),
            "X86Stdcall" => Ok(Conv::X86Stdcall),
            "X86ThisCall" => Ok(Conv::X86ThisCall),
            "X86VectorCall" => Ok(Conv::X86VectorCall),
            "X86_64SysV" => Ok(Conv::X86_64SysV),
            "X86_64Win64" => Ok(Conv::X86_64Win64),
            "AmdGpuKernel" => Ok(Conv::AmdGpuKernel),
            "AvrInterrupt" => Ok(Conv::AvrInterrupt),
            "AvrNonBlockingInterrupt" => Ok(Conv::AvrNonBlockingInterrupt),
            _ => Err(format!("'{s}' is not a valid value for entry function call convention.")),
        }
    }
}

// Some types are used a lot. Make sure they don't unintentionally get bigger.
#[cfg(all(target_arch = "x86_64", target_pointer_width = "64"))]
mod size_asserts {
    use super::*;
    use rustc_data_structures::static_assert_size;
    // tidy-alphabetical-start
    static_assert_size!(ArgAbi<'_, usize>, 56);
    static_assert_size!(FnAbi<'_, usize>, 80);
    // tidy-alphabetical-end
}

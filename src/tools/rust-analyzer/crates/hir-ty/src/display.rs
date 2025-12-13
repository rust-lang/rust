//! The `HirDisplay` trait, which serves two purposes: Turning various bits from
//! HIR back into source code, and just displaying them for debugging/testing
//! purposes.

use std::{
    fmt::{self, Debug},
    mem,
};

use base_db::Crate;
use either::Either;
use hir_def::{
    FindPathConfig, GenericDefId, HasModule, LocalFieldId, Lookup, ModuleDefId, ModuleId, TraitId,
    db::DefDatabase,
    expr_store::{ExpressionStore, path::Path},
    find_path::{self, PrefixKind},
    hir::generics::{TypeOrConstParamData, TypeParamProvenance, WherePredicate},
    item_scope::ItemInNs,
    item_tree::FieldsShape,
    lang_item::LangItems,
    signatures::VariantFields,
    type_ref::{
        ConstRef, LifetimeRef, LifetimeRefId, TraitBoundModifier, TypeBound, TypeRef, TypeRefId,
        UseArgRef,
    },
    visibility::Visibility,
};
use hir_expand::{mod_path::PathKind, name::Name};
use intern::{Internable, Interned, sym};
use itertools::Itertools;
use la_arena::ArenaMap;
use rustc_apfloat::{
    Float,
    ieee::{Half as f16, Quad as f128},
};
use rustc_ast_ir::FloatTy;
use rustc_hash::FxHashSet;
use rustc_type_ir::{
    AliasTyKind, BoundVarIndexKind, CoroutineArgsParts, CoroutineClosureArgsParts, RegionKind,
    Upcast,
    inherent::{AdtDef, GenericArgs as _, IntoKind, SliceLike, Term as _, Ty as _, Tys as _},
};
use smallvec::SmallVec;
use span::Edition;
use stdx::never;

use crate::{
    CallableDefId, FnAbi, ImplTraitId, InferenceResult, MemoryMap, ParamEnvAndCrate, consteval,
    db::{HirDatabase, InternedClosure, InternedCoroutine},
    generics::generics,
    layout::Layout,
    lower::GenericPredicates,
    mir::pad16,
    next_solver::{
        AliasTy, Clause, ClauseKind, Const, ConstKind, DbInterner, EarlyBinder,
        ExistentialPredicate, FnSig, GenericArg, GenericArgs, ParamEnv, PolyFnSig, Region,
        SolverDefId, Term, TraitRef, Ty, TyKind, TypingMode,
        abi::Safety,
        infer::{DbInternerInferExt, traits::ObligationCause},
    },
    primitive,
    utils::{detect_variant_from_bytes, fn_traits},
};

pub type Result<T = (), E = HirDisplayError> = std::result::Result<T, E>;

pub trait HirWrite: fmt::Write {
    fn start_location_link(&mut self, _location: ModuleDefId) {}
    fn end_location_link(&mut self) {}
}

// String will ignore link metadata
impl HirWrite for String {}

// `core::Formatter` will ignore metadata
impl HirWrite for fmt::Formatter<'_> {}

pub struct HirFormatter<'a, 'db> {
    /// The database handle
    pub db: &'db dyn HirDatabase,
    pub interner: DbInterner<'db>,
    /// The sink to write into
    fmt: &'a mut dyn HirWrite,
    /// A buffer to intercept writes with, this allows us to track the overall size of the formatted output.
    buf: String,
    /// The current size of the formatted output.
    curr_size: usize,
    /// Size from which we should truncate the output.
    max_size: Option<usize>,
    /// When rendering something that has a concept of "children" (like fields in a struct), this limits
    /// how many should be rendered.
    pub entity_limit: Option<usize>,
    /// When rendering functions, whether to show the constraint from the container
    show_container_bounds: bool,
    omit_verbose_types: bool,
    closure_style: ClosureStyle,
    display_lifetimes: DisplayLifetime,
    display_kind: DisplayKind,
    display_target: DisplayTarget,
    bounds_formatting_ctx: BoundsFormattingCtx<'db>,
}

// FIXME: To consider, ref and dyn trait lifetimes can be omitted if they are `'_`, path args should
// not be when in signatures
// So this enum does not encode this well enough
// Also 'static can be omitted for ref and dyn trait lifetimes in static/const item types
// FIXME: Also named lifetimes may be rendered in places where their name is not in scope?
#[derive(Copy, Clone)]
pub enum DisplayLifetime {
    Always,
    OnlyStatic,
    OnlyNamed,
    OnlyNamedOrStatic,
    Never,
}

#[derive(Default)]
enum BoundsFormattingCtx<'db> {
    Entered {
        /// We can have recursive bounds like the following case:
        /// ```ignore
        /// where
        ///     T: Foo,
        ///     T::FooAssoc: Baz<<T::FooAssoc as Bar>::BarAssoc> + Bar
        /// ```
        /// So, record the projection types met while formatting bounds and
        //. prevent recursing into their bounds to avoid infinite loops.
        projection_tys_met: FxHashSet<AliasTy<'db>>,
    },
    #[default]
    Exited,
}

impl<'db> BoundsFormattingCtx<'db> {
    fn contains(&self, proj: &AliasTy<'db>) -> bool {
        match self {
            BoundsFormattingCtx::Entered { projection_tys_met } => {
                projection_tys_met.contains(proj)
            }
            BoundsFormattingCtx::Exited => false,
        }
    }
}

impl<'db> HirFormatter<'_, 'db> {
    fn start_location_link(&mut self, location: ModuleDefId) {
        self.fmt.start_location_link(location);
    }

    fn end_location_link(&mut self) {
        self.fmt.end_location_link();
    }

    fn format_bounds_with<T, F: FnOnce(&mut Self) -> T>(
        &mut self,
        target: AliasTy<'db>,
        format_bounds: F,
    ) -> T {
        match self.bounds_formatting_ctx {
            BoundsFormattingCtx::Entered { ref mut projection_tys_met } => {
                projection_tys_met.insert(target);
                format_bounds(self)
            }
            BoundsFormattingCtx::Exited => {
                let mut projection_tys_met = FxHashSet::default();
                projection_tys_met.insert(target);
                self.bounds_formatting_ctx = BoundsFormattingCtx::Entered { projection_tys_met };
                let res = format_bounds(self);
                // Since we want to prevent only the infinite recursions in bounds formatting
                // and do not want to skip formatting of other separate bounds, clear context
                // when exiting the formatting of outermost bounds
                self.bounds_formatting_ctx = BoundsFormattingCtx::Exited;
                res
            }
        }
    }

    fn render_region(&self, lifetime: Region<'db>) -> bool {
        match self.display_lifetimes {
            DisplayLifetime::Always => true,
            DisplayLifetime::OnlyStatic => matches!(lifetime.kind(), RegionKind::ReStatic),
            DisplayLifetime::OnlyNamed => {
                matches!(lifetime.kind(), RegionKind::ReEarlyParam(_))
            }
            DisplayLifetime::OnlyNamedOrStatic => {
                matches!(lifetime.kind(), RegionKind::ReStatic | RegionKind::ReEarlyParam(_))
            }
            DisplayLifetime::Never => false,
        }
    }
}

pub trait HirDisplay<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result;

    /// Returns a `Display`able type that is human-readable.
    fn into_displayable<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        max_size: Option<usize>,
        limited_size: Option<usize>,
        omit_verbose_types: bool,
        display_target: DisplayTarget,
        display_kind: DisplayKind,
        closure_style: ClosureStyle,
        show_container_bounds: bool,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        assert!(
            !matches!(display_kind, DisplayKind::SourceCode { .. }),
            "HirDisplayWrapper cannot fail with DisplaySourceCodeError, use HirDisplay::hir_fmt directly instead"
        );
        HirDisplayWrapper {
            db,
            t: self,
            max_size,
            limited_size,
            omit_verbose_types,
            display_target,
            display_kind,
            closure_style,
            show_container_bounds,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
        }
    }

    /// Returns a `Display`able type that is human-readable.
    /// Use this for showing types to the user (e.g. diagnostics)
    fn display<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            limited_size: None,
            omit_verbose_types: false,
            closure_style: ClosureStyle::ImplFn,
            display_target,
            display_kind: DisplayKind::Diagnostics,
            show_container_bounds: false,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
        }
    }

    /// Returns a `Display`able type that is human-readable and tries to be succinct.
    /// Use this for showing types to the user where space is constrained (e.g. doc popups)
    fn display_truncated<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        max_size: Option<usize>,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size,
            limited_size: None,
            omit_verbose_types: true,
            closure_style: ClosureStyle::ImplFn,
            display_target,
            display_kind: DisplayKind::Diagnostics,
            show_container_bounds: false,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
        }
    }

    /// Returns a `Display`able type that is human-readable and tries to limit the number of items inside.
    /// Use this for showing definitions which may contain too many items, like `trait`, `struct`, `enum`
    fn display_limited<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        limited_size: Option<usize>,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            limited_size,
            omit_verbose_types: true,
            closure_style: ClosureStyle::ImplFn,
            display_target,
            display_kind: DisplayKind::Diagnostics,
            show_container_bounds: false,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
        }
    }

    /// Returns a String representation of `self` that can be inserted into the given module.
    /// Use this when generating code (e.g. assists)
    fn display_source_code<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        module_id: ModuleId,
        allow_opaque: bool,
    ) -> Result<String, DisplaySourceCodeError> {
        let mut result = String::new();
        let interner = DbInterner::new_with(db, module_id.krate(db));
        match self.hir_fmt(&mut HirFormatter {
            db,
            interner,
            fmt: &mut result,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: None,
            entity_limit: None,
            omit_verbose_types: false,
            closure_style: ClosureStyle::ImplFn,
            display_target: DisplayTarget::from_crate(db, module_id.krate(db)),
            display_kind: DisplayKind::SourceCode { target_module_id: module_id, allow_opaque },
            show_container_bounds: false,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
            bounds_formatting_ctx: Default::default(),
        }) {
            Ok(()) => {}
            Err(HirDisplayError::FmtError) => panic!("Writing to String can't fail!"),
            Err(HirDisplayError::DisplaySourceCodeError(e)) => return Err(e),
        };
        Ok(result)
    }

    /// Returns a String representation of `self` for test purposes
    fn display_test<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            limited_size: None,
            omit_verbose_types: false,
            closure_style: ClosureStyle::ImplFn,
            display_target,
            display_kind: DisplayKind::Test,
            show_container_bounds: false,
            display_lifetimes: DisplayLifetime::Always,
        }
    }

    /// Returns a String representation of `self` that shows the constraint from
    /// the container for functions
    fn display_with_container_bounds<'a>(
        &'a self,
        db: &'db dyn HirDatabase,
        show_container_bounds: bool,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, 'db, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            limited_size: None,
            omit_verbose_types: false,
            closure_style: ClosureStyle::ImplFn,
            display_target,
            display_kind: DisplayKind::Diagnostics,
            show_container_bounds,
            display_lifetimes: DisplayLifetime::OnlyNamedOrStatic,
        }
    }
}

impl<'db> HirFormatter<'_, 'db> {
    pub fn krate(&self) -> Crate {
        self.display_target.krate
    }

    pub fn edition(&self) -> Edition {
        self.display_target.edition
    }

    #[inline]
    pub fn lang_items(&self) -> &'db LangItems {
        self.interner.lang_items()
    }

    pub fn write_joined<T: HirDisplay<'db>>(
        &mut self,
        iter: impl IntoIterator<Item = T>,
        sep: &str,
    ) -> Result {
        let mut first = true;
        for e in iter {
            if !first {
                write!(self, "{sep}")?;
            }
            first = false;

            // Abbreviate multiple omitted types with a single ellipsis.
            if self.should_truncate() {
                return write!(self, "{TYPE_HINT_TRUNCATION}");
            }

            e.hir_fmt(self)?;
        }
        Ok(())
    }

    /// This allows using the `write!` macro directly with a `HirFormatter`.
    pub fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> Result {
        // We write to a buffer first to track output size
        self.buf.clear();
        fmt::write(&mut self.buf, args)?;
        self.curr_size += self.buf.len();

        // Then we write to the internal formatter from the buffer
        self.fmt.write_str(&self.buf).map_err(HirDisplayError::from)
    }

    pub fn write_str(&mut self, s: &str) -> Result {
        self.fmt.write_str(s)?;
        Ok(())
    }

    pub fn write_char(&mut self, c: char) -> Result {
        self.fmt.write_char(c)?;
        Ok(())
    }

    pub fn should_truncate(&self) -> bool {
        match self.max_size {
            Some(max_size) => self.curr_size >= max_size,
            None => false,
        }
    }

    pub fn omit_verbose_types(&self) -> bool {
        self.omit_verbose_types
    }

    pub fn show_container_bounds(&self) -> bool {
        self.show_container_bounds
    }
}

#[derive(Debug, Clone, Copy)]
pub struct DisplayTarget {
    krate: Crate,
    pub edition: Edition,
}

impl DisplayTarget {
    pub fn from_crate(db: &dyn HirDatabase, krate: Crate) -> Self {
        let edition = krate.data(db).edition;
        Self { krate, edition }
    }
}

#[derive(Clone, Copy)]
pub enum DisplayKind {
    /// Display types for inlays, doc popups, autocompletion, etc...
    /// Showing `{unknown}` or not qualifying paths is fine here.
    /// There's no reason for this to fail.
    Diagnostics,
    /// Display types for inserting them in source files.
    /// The generated code should compile, so paths need to be qualified.
    SourceCode { target_module_id: ModuleId, allow_opaque: bool },
    /// Only for test purpose to keep real types
    Test,
}

impl DisplayKind {
    fn is_source_code(self) -> bool {
        matches!(self, Self::SourceCode { .. })
    }

    fn allows_opaque(self) -> bool {
        match self {
            Self::SourceCode { allow_opaque, .. } => allow_opaque,
            _ => true,
        }
    }
}

#[derive(Debug)]
pub enum DisplaySourceCodeError {
    PathNotFound,
    Coroutine,
    OpaqueType,
}

pub enum HirDisplayError {
    /// Errors that can occur when generating source code
    DisplaySourceCodeError(DisplaySourceCodeError),
    /// `FmtError` is required to be compatible with std::fmt::Display
    FmtError,
}
impl From<fmt::Error> for HirDisplayError {
    fn from(_: fmt::Error) -> Self {
        Self::FmtError
    }
}

pub struct HirDisplayWrapper<'a, 'db, T> {
    db: &'db dyn HirDatabase,
    t: &'a T,
    max_size: Option<usize>,
    limited_size: Option<usize>,
    omit_verbose_types: bool,
    closure_style: ClosureStyle,
    display_kind: DisplayKind,
    display_target: DisplayTarget,
    show_container_bounds: bool,
    display_lifetimes: DisplayLifetime,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ClosureStyle {
    /// `impl FnX(i32, i32) -> i32`, where `FnX` is the most special trait between `Fn`, `FnMut`, `FnOnce` that the
    /// closure implements. This is the default.
    ImplFn,
    /// `|i32, i32| -> i32`
    RANotation,
    /// `{closure#14825}`, useful for some diagnostics (like type mismatch) and internal usage.
    ClosureWithId,
    /// `{closure#14825}<i32, ()>`, useful for internal usage.
    ClosureWithSubst,
    /// `…`, which is the `TYPE_HINT_TRUNCATION`
    Hide,
}

impl<'db, T: HirDisplay<'db>> HirDisplayWrapper<'_, 'db, T> {
    pub fn write_to<F: HirWrite>(&self, f: &mut F) -> Result {
        let krate = self.display_target.krate;
        let interner = DbInterner::new_with(self.db, krate);
        self.t.hir_fmt(&mut HirFormatter {
            db: self.db,
            interner,
            fmt: f,
            buf: String::with_capacity(self.max_size.unwrap_or(20)),
            curr_size: 0,
            max_size: self.max_size,
            entity_limit: self.limited_size,
            omit_verbose_types: self.omit_verbose_types,
            display_kind: self.display_kind,
            display_target: self.display_target,
            closure_style: self.closure_style,
            show_container_bounds: self.show_container_bounds,
            display_lifetimes: self.display_lifetimes,
            bounds_formatting_ctx: Default::default(),
        })
    }

    pub fn with_closure_style(mut self, c: ClosureStyle) -> Self {
        self.closure_style = c;
        self
    }

    pub fn with_lifetime_display(mut self, l: DisplayLifetime) -> Self {
        self.display_lifetimes = l;
        self
    }
}

impl<'db, T> fmt::Display for HirDisplayWrapper<'_, 'db, T>
where
    T: HirDisplay<'db>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.write_to(f) {
            Ok(()) => Ok(()),
            Err(HirDisplayError::FmtError) => Err(fmt::Error),
            Err(HirDisplayError::DisplaySourceCodeError(_)) => {
                // This should never happen
                panic!(
                    "HirDisplay::hir_fmt failed with DisplaySourceCodeError when calling Display::fmt!"
                )
            }
        }
    }
}

const TYPE_HINT_TRUNCATION: &str = "…";

impl<'db, T: HirDisplay<'db>> HirDisplay<'db> for &T {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl<'db, T: HirDisplay<'db> + Internable> HirDisplay<'db> for Interned<T> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        HirDisplay::hir_fmt(self.as_ref(), f)
    }
}

fn write_projection<'db>(f: &mut HirFormatter<'_, 'db>, alias: &AliasTy<'db>) -> Result {
    if f.should_truncate() {
        return write!(f, "{TYPE_HINT_TRUNCATION}");
    }
    let trait_ref = alias.trait_ref(f.interner);
    let self_ty = trait_ref.self_ty();

    // if we are projection on a type parameter, check if the projection target has bounds
    // itself, if so, we render them directly as `impl Bound` instead of the less useful
    // `<Param as Trait>::Assoc`
    if !f.display_kind.is_source_code()
        && let TyKind::Param(param) = self_ty.kind()
        && !f.bounds_formatting_ctx.contains(alias)
    {
        // FIXME: We shouldn't use `param.id`, it should be removed. We should know the
        // `GenericDefId` from the formatted type (store it inside the `HirFormatter`).
        let bounds = GenericPredicates::query_all(f.db, param.id.parent())
            .iter_identity_copied()
            .filter(|wc| {
                let ty = match wc.kind().skip_binder() {
                    ClauseKind::Trait(tr) => tr.self_ty(),
                    ClauseKind::TypeOutlives(t) => t.0,
                    _ => return false,
                };
                let TyKind::Alias(AliasTyKind::Projection, a) = ty.kind() else {
                    return false;
                };
                a == *alias
            })
            .collect::<Vec<_>>();
        if !bounds.is_empty() {
            return f.format_bounds_with(*alias, |f| {
                write_bounds_like_dyn_trait_with_prefix(
                    f,
                    "impl",
                    Either::Left(Ty::new_alias(f.interner, AliasTyKind::Projection, *alias)),
                    &bounds,
                    SizedByDefault::NotSized,
                )
            });
        }
    }

    write!(f, "<")?;
    self_ty.hir_fmt(f)?;
    write!(f, " as ")?;
    trait_ref.hir_fmt(f)?;
    write!(
        f,
        ">::{}",
        f.db.type_alias_signature(alias.def_id.expect_type_alias()).name.display(f.db, f.edition())
    )?;
    let proj_params = &alias.args.as_slice()[trait_ref.args.len()..];
    hir_fmt_generics(f, proj_params, None, None)
}

impl<'db> HirDisplay<'db> for GenericArg<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        match self {
            GenericArg::Ty(ty) => ty.hir_fmt(f),
            GenericArg::Lifetime(lt) => lt.hir_fmt(f),
            GenericArg::Const(c) => c.hir_fmt(f),
        }
    }
}

impl<'db> HirDisplay<'db> for Const<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        match self.kind() {
            ConstKind::Placeholder(_) => write!(f, "<placeholder>"),
            ConstKind::Bound(BoundVarIndexKind::Bound(db), bound_const) => {
                write!(f, "?{}.{}", db.as_u32(), bound_const.var.as_u32())
            }
            ConstKind::Bound(BoundVarIndexKind::Canonical, bound_const) => {
                write!(f, "?c.{}", bound_const.var.as_u32())
            }
            ConstKind::Infer(..) => write!(f, "#c#"),
            ConstKind::Param(param) => {
                let generics = generics(f.db, param.id.parent());
                let param_data = &generics[param.id.local_id()];
                write!(f, "{}", param_data.name().unwrap().display(f.db, f.edition()))?;
                Ok(())
            }
            ConstKind::Value(const_bytes) => render_const_scalar(
                f,
                &const_bytes.value.inner().memory,
                &const_bytes.value.inner().memory_map,
                const_bytes.ty,
            ),
            ConstKind::Unevaluated(unev) => {
                let c = unev.def.0;
                write!(f, "{}", c.name(f.db))?;
                hir_fmt_generics(f, unev.args.as_slice(), c.generic_def(f.db), None)?;
                Ok(())
            }
            ConstKind::Error(..) => f.write_char('_'),
            ConstKind::Expr(..) => write!(f, "<const-expr>"),
        }
    }
}

fn render_const_scalar<'db>(
    f: &mut HirFormatter<'_, 'db>,
    b: &[u8],
    memory_map: &MemoryMap<'db>,
    ty: Ty<'db>,
) -> Result {
    let param_env = ParamEnv::empty();
    let infcx = f.interner.infer_ctxt().build(TypingMode::PostAnalysis);
    let ty = infcx.at(&ObligationCause::new(), param_env).deeply_normalize(ty).unwrap_or(ty);
    render_const_scalar_inner(f, b, memory_map, ty, param_env)
}

fn render_const_scalar_inner<'db>(
    f: &mut HirFormatter<'_, 'db>,
    b: &[u8],
    memory_map: &MemoryMap<'db>,
    ty: Ty<'db>,
    param_env: ParamEnv<'db>,
) -> Result {
    use TyKind;
    let param_env = ParamEnvAndCrate { param_env, krate: f.krate() };
    match ty.kind() {
        TyKind::Bool => write!(f, "{}", b[0] != 0),
        TyKind::Char => {
            let it = u128::from_le_bytes(pad16(b, false)) as u32;
            let Ok(c) = char::try_from(it) else {
                return f.write_str("<unicode-error>");
            };
            write!(f, "{c:?}")
        }
        TyKind::Int(_) => {
            let it = i128::from_le_bytes(pad16(b, true));
            write!(f, "{it}")
        }
        TyKind::Uint(_) => {
            let it = u128::from_le_bytes(pad16(b, false));
            write!(f, "{it}")
        }
        TyKind::Float(fl) => match fl {
            FloatTy::F16 => {
                // FIXME(#17451): Replace with builtins once they are stabilised.
                let it = f16::from_bits(u16::from_le_bytes(b.try_into().unwrap()).into());
                let s = it.to_string();
                if s.strip_prefix('-').unwrap_or(&s).chars().all(|c| c.is_ascii_digit()) {
                    // Match Rust debug formatting
                    write!(f, "{s}.0")
                } else {
                    write!(f, "{s}")
                }
            }
            FloatTy::F32 => {
                let it = f32::from_le_bytes(b.try_into().unwrap());
                write!(f, "{it:?}")
            }
            FloatTy::F64 => {
                let it = f64::from_le_bytes(b.try_into().unwrap());
                write!(f, "{it:?}")
            }
            FloatTy::F128 => {
                // FIXME(#17451): Replace with builtins once they are stabilised.
                let it = f128::from_bits(u128::from_le_bytes(b.try_into().unwrap()));
                let s = it.to_string();
                if s.strip_prefix('-').unwrap_or(&s).chars().all(|c| c.is_ascii_digit()) {
                    // Match Rust debug formatting
                    write!(f, "{s}.0")
                } else {
                    write!(f, "{s}")
                }
            }
        },
        TyKind::Ref(_, t, _) => match t.kind() {
            TyKind::Str => {
                let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
                let size = usize::from_le_bytes(b[b.len() / 2..].try_into().unwrap());
                let Some(bytes) = memory_map.get(addr, size) else {
                    return f.write_str("<ref-data-not-available>");
                };
                let s = std::str::from_utf8(bytes).unwrap_or("<utf8-error>");
                write!(f, "{s:?}")
            }
            TyKind::Slice(ty) => {
                let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
                let count = usize::from_le_bytes(b[b.len() / 2..].try_into().unwrap());
                let Ok(layout) = f.db.layout_of_ty(ty, param_env) else {
                    return f.write_str("<layout-error>");
                };
                let size_one = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size_one * count) else {
                    return f.write_str("<ref-data-not-available>");
                };
                let expected_len = count * size_one;
                if bytes.len() < expected_len {
                    never!(
                        "Memory map size is too small. Expected {expected_len}, got {}",
                        bytes.len(),
                    );
                    return f.write_str("<layout-error>");
                }
                f.write_str("&[")?;
                let mut first = true;
                for i in 0..count {
                    if first {
                        first = false;
                    } else {
                        f.write_str(", ")?;
                    }
                    let offset = size_one * i;
                    render_const_scalar(f, &bytes[offset..offset + size_one], memory_map, ty)?;
                }
                f.write_str("]")
            }
            TyKind::Dynamic(_, _) => {
                let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
                let ty_id = usize::from_le_bytes(b[b.len() / 2..].try_into().unwrap());
                let Ok(t) = memory_map.vtable_ty(ty_id) else {
                    return f.write_str("<ty-missing-in-vtable-map>");
                };
                let Ok(layout) = f.db.layout_of_ty(t, param_env) else {
                    return f.write_str("<layout-error>");
                };
                let size = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size) else {
                    return f.write_str("<ref-data-not-available>");
                };
                f.write_str("&")?;
                render_const_scalar(f, bytes, memory_map, t)
            }
            TyKind::Adt(adt, _) if b.len() == 2 * size_of::<usize>() => match adt.def_id().0 {
                hir_def::AdtId::StructId(s) => {
                    let data = f.db.struct_signature(s);
                    write!(f, "&{}", data.name.display(f.db, f.edition()))?;
                    Ok(())
                }
                _ => f.write_str("<unsized-enum-or-union>"),
            },
            _ => {
                let addr = usize::from_le_bytes(match b.try_into() {
                    Ok(b) => b,
                    Err(_) => {
                        never!(
                            "tried rendering ty {:?} in const ref with incorrect byte count {}",
                            t,
                            b.len()
                        );
                        return f.write_str("<layout-error>");
                    }
                });
                let Ok(layout) = f.db.layout_of_ty(t, param_env) else {
                    return f.write_str("<layout-error>");
                };
                let size = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size) else {
                    return f.write_str("<ref-data-not-available>");
                };
                f.write_str("&")?;
                render_const_scalar(f, bytes, memory_map, t)
            }
        },
        TyKind::Tuple(tys) => {
            let Ok(layout) = f.db.layout_of_ty(ty, param_env) else {
                return f.write_str("<layout-error>");
            };
            f.write_str("(")?;
            let mut first = true;
            for (id, ty) in tys.iter().enumerate() {
                if first {
                    first = false;
                } else {
                    f.write_str(", ")?;
                }
                let offset = layout.fields.offset(id).bytes_usize();
                let Ok(layout) = f.db.layout_of_ty(ty, param_env) else {
                    f.write_str("<layout-error>")?;
                    continue;
                };
                let size = layout.size.bytes_usize();
                render_const_scalar(f, &b[offset..offset + size], memory_map, ty)?;
            }
            f.write_str(")")
        }
        TyKind::Adt(def, args) => {
            let def = def.def_id().0;
            let Ok(layout) = f.db.layout_of_adt(def, args, param_env) else {
                return f.write_str("<layout-error>");
            };
            match def {
                hir_def::AdtId::StructId(s) => {
                    let data = f.db.struct_signature(s);
                    write!(f, "{}", data.name.display(f.db, f.edition()))?;
                    let field_types = f.db.field_types(s.into());
                    render_variant_after_name(
                        s.fields(f.db),
                        f,
                        &field_types,
                        f.db.trait_environment(def.into()),
                        &layout,
                        args,
                        b,
                        memory_map,
                    )
                }
                hir_def::AdtId::UnionId(u) => {
                    write!(f, "{}", f.db.union_signature(u).name.display(f.db, f.edition()))
                }
                hir_def::AdtId::EnumId(e) => {
                    let Ok(target_data_layout) = f.db.target_data_layout(f.krate()) else {
                        return f.write_str("<target-layout-not-available>");
                    };
                    let Some((var_id, var_layout)) =
                        detect_variant_from_bytes(&layout, f.db, &target_data_layout, b, e)
                    else {
                        return f.write_str("<failed-to-detect-variant>");
                    };
                    let loc = var_id.lookup(f.db);
                    write!(
                        f,
                        "{}",
                        loc.parent.enum_variants(f.db).variants[loc.index as usize]
                            .1
                            .display(f.db, f.edition())
                    )?;
                    let field_types = f.db.field_types(var_id.into());
                    render_variant_after_name(
                        var_id.fields(f.db),
                        f,
                        &field_types,
                        f.db.trait_environment(def.into()),
                        var_layout,
                        args,
                        b,
                        memory_map,
                    )
                }
            }
        }
        TyKind::FnDef(..) => ty.hir_fmt(f),
        TyKind::FnPtr(_, _) | TyKind::RawPtr(_, _) => {
            let it = u128::from_le_bytes(pad16(b, false));
            write!(f, "{it:#X} as ")?;
            ty.hir_fmt(f)
        }
        TyKind::Array(ty, len) => {
            let Some(len) = consteval::try_const_usize(f.db, len) else {
                return f.write_str("<unknown-array-len>");
            };
            let Ok(layout) = f.db.layout_of_ty(ty, param_env) else {
                return f.write_str("<layout-error>");
            };
            let size_one = layout.size.bytes_usize();
            f.write_str("[")?;
            let mut first = true;
            for i in 0..len as usize {
                if first {
                    first = false;
                } else {
                    f.write_str(", ")?;
                }
                let offset = size_one * i;
                render_const_scalar(f, &b[offset..offset + size_one], memory_map, ty)?;
            }
            f.write_str("]")
        }
        TyKind::Never => f.write_str("!"),
        TyKind::Closure(_, _) => f.write_str("<closure>"),
        TyKind::Coroutine(_, _) => f.write_str("<coroutine>"),
        TyKind::CoroutineWitness(_, _) => f.write_str("<coroutine-witness>"),
        TyKind::CoroutineClosure(_, _) => f.write_str("<coroutine-closure>"),
        TyKind::UnsafeBinder(_) => f.write_str("<unsafe-binder>"),
        // The below arms are unreachable, since const eval will bail out before here.
        TyKind::Foreign(_) => f.write_str("<extern-type>"),
        TyKind::Pat(_, _) => f.write_str("<pat>"),
        TyKind::Error(..)
        | TyKind::Placeholder(_)
        | TyKind::Alias(_, _)
        | TyKind::Param(_)
        | TyKind::Bound(_, _)
        | TyKind::Infer(_) => f.write_str("<placeholder-or-unknown-type>"),
        // The below arms are unreachable, since we handled them in ref case.
        TyKind::Slice(_) | TyKind::Str | TyKind::Dynamic(_, _) => f.write_str("<unsized-value>"),
    }
}

fn render_variant_after_name<'db>(
    data: &VariantFields,
    f: &mut HirFormatter<'_, 'db>,
    field_types: &ArenaMap<LocalFieldId, EarlyBinder<'db, Ty<'db>>>,
    param_env: ParamEnv<'db>,
    layout: &Layout,
    args: GenericArgs<'db>,
    b: &[u8],
    memory_map: &MemoryMap<'db>,
) -> Result {
    let param_env = ParamEnvAndCrate { param_env, krate: f.krate() };
    match data.shape {
        FieldsShape::Record | FieldsShape::Tuple => {
            let render_field = |f: &mut HirFormatter<'_, 'db>, id: LocalFieldId| {
                let offset = layout.fields.offset(u32::from(id.into_raw()) as usize).bytes_usize();
                let ty = field_types[id].instantiate(f.interner, args);
                let Ok(layout) = f.db.layout_of_ty(ty, param_env) else {
                    return f.write_str("<layout-error>");
                };
                let size = layout.size.bytes_usize();
                render_const_scalar(f, &b[offset..offset + size], memory_map, ty)
            };
            let mut it = data.fields().iter();
            if matches!(data.shape, FieldsShape::Record) {
                write!(f, " {{")?;
                if let Some((id, data)) = it.next() {
                    write!(f, " {}: ", data.name.display(f.db, f.edition()))?;
                    render_field(f, id)?;
                }
                for (id, data) in it {
                    write!(f, ", {}: ", data.name.display(f.db, f.edition()))?;
                    render_field(f, id)?;
                }
                write!(f, " }}")?;
            } else {
                let mut it = it.map(|it| it.0);
                write!(f, "(")?;
                if let Some(id) = it.next() {
                    render_field(f, id)?;
                }
                for id in it {
                    write!(f, ", ")?;
                    render_field(f, id)?;
                }
                write!(f, ")")?;
            }
            Ok(())
        }
        FieldsShape::Unit => Ok(()),
    }
}

impl<'db> HirDisplay<'db> for Ty<'db> {
    fn hir_fmt(&self, f @ &mut HirFormatter { db, .. }: &mut HirFormatter<'_, 'db>) -> Result {
        let interner = f.interner;
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }

        use TyKind;
        match self.kind() {
            TyKind::Never => write!(f, "!")?,
            TyKind::Str => write!(f, "str")?,
            TyKind::Bool => write!(f, "bool")?,
            TyKind::Char => write!(f, "char")?,
            TyKind::Float(t) => write!(f, "{}", primitive::float_ty_to_string(t))?,
            TyKind::Int(t) => write!(f, "{}", primitive::int_ty_to_string(t))?,
            TyKind::Uint(t) => write!(f, "{}", primitive::uint_ty_to_string(t))?,
            TyKind::Slice(t) => {
                write!(f, "[")?;
                t.hir_fmt(f)?;
                write!(f, "]")?;
            }
            TyKind::Array(t, c) => {
                write!(f, "[")?;
                t.hir_fmt(f)?;
                write!(f, "; ")?;
                c.hir_fmt(f)?;
                write!(f, "]")?;
            }
            kind @ (TyKind::RawPtr(t, m) | TyKind::Ref(_, t, m)) => {
                if let TyKind::Ref(l, _, _) = kind {
                    f.write_char('&')?;
                    if f.render_region(l) {
                        l.hir_fmt(f)?;
                        f.write_char(' ')?;
                    }
                    match m {
                        rustc_ast_ir::Mutability::Not => (),
                        rustc_ast_ir::Mutability::Mut => f.write_str("mut ")?,
                    }
                } else {
                    write!(
                        f,
                        "*{}",
                        match m {
                            rustc_ast_ir::Mutability::Not => "const ",
                            rustc_ast_ir::Mutability::Mut => "mut ",
                        }
                    )?;
                }

                // FIXME: all this just to decide whether to use parentheses...
                let (preds_to_print, has_impl_fn_pred) = match t.kind() {
                    TyKind::Dynamic(bounds, region) => {
                        let contains_impl_fn =
                            bounds.iter().any(|bound| match bound.skip_binder() {
                                ExistentialPredicate::Trait(trait_ref) => {
                                    let trait_ = trait_ref.def_id.0;
                                    fn_traits(f.lang_items()).any(|it| it == trait_)
                                }
                                _ => false,
                            });
                        let render_lifetime = f.render_region(region);
                        (bounds.len() + render_lifetime as usize, contains_impl_fn)
                    }
                    TyKind::Alias(AliasTyKind::Opaque, ty) => {
                        let opaque_ty_id = match ty.def_id {
                            SolverDefId::InternedOpaqueTyId(id) => id,
                            _ => unreachable!(),
                        };
                        let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty_id);
                        if let ImplTraitId::ReturnTypeImplTrait(func, _) = impl_trait_id {
                            let data = impl_trait_id.predicates(db);
                            let bounds =
                                || data.iter_instantiated_copied(f.interner, ty.args.as_slice());
                            let mut len = bounds().count();

                            // Don't count Sized but count when it absent
                            // (i.e. when explicit ?Sized bound is set).
                            let default_sized = SizedByDefault::Sized { anchor: func.krate(db) };
                            let sized_bounds = bounds()
                                .filter(|b| {
                                    matches!(
                                        b.kind().skip_binder(),
                                        ClauseKind::Trait(trait_ref)
                                            if default_sized.is_sized_trait(
                                                trait_ref.def_id().0,
                                                db,
                                            ),
                                    )
                                })
                                .count();
                            match sized_bounds {
                                0 => len += 1,
                                _ => {
                                    len = len.saturating_sub(sized_bounds);
                                }
                            }

                            let contains_impl_fn = bounds().any(|bound| {
                                if let ClauseKind::Trait(trait_ref) = bound.kind().skip_binder() {
                                    let trait_ = trait_ref.def_id().0;
                                    fn_traits(f.lang_items()).any(|it| it == trait_)
                                } else {
                                    false
                                }
                            });
                            (len, contains_impl_fn)
                        } else {
                            (0, false)
                        }
                    }
                    _ => (0, false),
                };

                if has_impl_fn_pred && preds_to_print <= 2 {
                    return t.hir_fmt(f);
                }

                if preds_to_print > 1 {
                    write!(f, "(")?;
                    t.hir_fmt(f)?;
                    write!(f, ")")?;
                } else {
                    t.hir_fmt(f)?;
                }
            }
            TyKind::Tuple(tys) => {
                if tys.len() == 1 {
                    write!(f, "(")?;
                    tys.as_slice()[0].hir_fmt(f)?;
                    write!(f, ",)")?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(tys.as_slice(), ", ")?;
                    write!(f, ")")?;
                }
            }
            TyKind::FnPtr(sig, header) => {
                let sig = sig.with(header);
                sig.hir_fmt(f)?;
            }
            TyKind::FnDef(def, args) => {
                let def = def.0;
                let sig = db.callable_item_signature(def).instantiate(interner, args);

                if f.display_kind.is_source_code() {
                    // `FnDef` is anonymous and there's no surface syntax for it. Show it as a
                    // function pointer type.
                    return sig.hir_fmt(f);
                }
                if let Safety::Unsafe = sig.safety() {
                    write!(f, "unsafe ")?;
                }
                if !matches!(sig.abi(), FnAbi::Rust | FnAbi::RustCall) {
                    f.write_str("extern \"")?;
                    f.write_str(sig.abi().as_str())?;
                    f.write_str("\" ")?;
                }

                let sig = sig.skip_binder();
                write!(f, "fn ")?;
                f.start_location_link(def.into());
                match def {
                    CallableDefId::FunctionId(ff) => {
                        write!(f, "{}", db.function_signature(ff).name.display(f.db, f.edition()))?
                    }
                    CallableDefId::StructId(s) => {
                        write!(f, "{}", db.struct_signature(s).name.display(f.db, f.edition()))?
                    }
                    CallableDefId::EnumVariantId(e) => {
                        let loc = e.lookup(db);
                        write!(
                            f,
                            "{}",
                            loc.parent.enum_variants(db).variants[loc.index as usize]
                                .1
                                .display(db, f.edition())
                        )?
                    }
                };
                f.end_location_link();

                if args.len() > 0 {
                    let generic_def_id = GenericDefId::from_callable(db, def);
                    let generics = generics(db, generic_def_id);
                    let (parent_len, self_param, type_, const_, impl_, lifetime) =
                        generics.provenance_split();
                    let parameters = args.as_slice();
                    debug_assert_eq!(
                        parameters.len(),
                        parent_len + self_param as usize + type_ + const_ + impl_ + lifetime
                    );
                    // We print all params except implicit impl Trait params. Still a bit weird; should we leave out parent and self?
                    if parameters.len() - impl_ > 0 {
                        let params_len = parameters.len();
                        // `parameters` are in the order of fn's params (including impl traits), fn's lifetimes
                        let parameters =
                            generic_args_sans_defaults(f, Some(generic_def_id), parameters);
                        assert!(params_len >= parameters.len());
                        let defaults = params_len - parameters.len();

                        // Normally, functions cannot have default parameters, but they can,
                        // for function-like things such as struct names or enum variants.
                        // The former cannot have defaults but does have parents,
                        // but the latter cannot have parents but can have defaults.
                        //
                        // However, it's also true that *traits* can have defaults too.
                        // In this case, there can be no function params.
                        let parent_end = if parent_len > 0 {
                            // If `parent_len` > 0, then there cannot be defaults on the function
                            // and all defaults must come from the parent.
                            parent_len - defaults
                        } else {
                            parent_len
                        };
                        let fn_params_no_impl_or_defaults = parameters.len() - parent_end - impl_;
                        let (parent_params, fn_params) = parameters.split_at(parent_end);

                        write!(f, "<")?;
                        hir_fmt_generic_arguments(f, parent_params, None)?;
                        if !parent_params.is_empty() && !fn_params.is_empty() {
                            write!(f, ", ")?;
                        }
                        hir_fmt_generic_arguments(
                            f,
                            &fn_params[..fn_params_no_impl_or_defaults],
                            None,
                        )?;
                        write!(f, ">")?;
                    }
                }
                write!(f, "(")?;
                f.write_joined(sig.inputs(), ", ")?;
                write!(f, ")")?;
                let ret = sig.output();
                if !ret.is_unit() {
                    write!(f, " -> ")?;
                    ret.hir_fmt(f)?;
                }
            }
            TyKind::Adt(def, parameters) => {
                let def_id = def.def_id().0;
                f.start_location_link(def_id.into());
                match f.display_kind {
                    DisplayKind::Diagnostics | DisplayKind::Test => {
                        let name = match def_id {
                            hir_def::AdtId::StructId(it) => db.struct_signature(it).name.clone(),
                            hir_def::AdtId::UnionId(it) => db.union_signature(it).name.clone(),
                            hir_def::AdtId::EnumId(it) => db.enum_signature(it).name.clone(),
                        };
                        write!(f, "{}", name.display(f.db, f.edition()))?;
                    }
                    DisplayKind::SourceCode { target_module_id: module_id, allow_opaque: _ } => {
                        if let Some(path) = find_path::find_path(
                            db,
                            ItemInNs::Types(def_id.into()),
                            module_id,
                            PrefixKind::Plain,
                            false,
                            // FIXME: no_std Cfg?
                            FindPathConfig {
                                prefer_no_std: false,
                                prefer_prelude: true,
                                prefer_absolute: false,
                                allow_unstable: true,
                            },
                        ) {
                            write!(f, "{}", path.display(f.db, f.edition()))?;
                        } else {
                            return Err(HirDisplayError::DisplaySourceCodeError(
                                DisplaySourceCodeError::PathNotFound,
                            ));
                        }
                    }
                }
                f.end_location_link();

                hir_fmt_generics(f, parameters.as_slice(), Some(def.def_id().0.into()), None)?;
            }
            TyKind::Alias(AliasTyKind::Projection, alias_ty) => write_projection(f, &alias_ty)?,
            TyKind::Foreign(alias) => {
                let type_alias = db.type_alias_signature(alias.0);
                f.start_location_link(alias.0.into());
                write!(f, "{}", type_alias.name.display(f.db, f.edition()))?;
                f.end_location_link();
            }
            TyKind::Alias(AliasTyKind::Opaque, alias_ty) => {
                let opaque_ty_id = match alias_ty.def_id {
                    SolverDefId::InternedOpaqueTyId(id) => id,
                    _ => unreachable!(),
                };
                if !f.display_kind.allows_opaque() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::OpaqueType,
                    ));
                }
                let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty_id);
                let data = impl_trait_id.predicates(db);
                let bounds = data
                    .iter_instantiated_copied(interner, alias_ty.args.as_slice())
                    .collect::<Vec<_>>();
                let krate = match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, _) => {
                        func.krate(db)
                        // FIXME: it would maybe be good to distinguish this from the alias type (when debug printing), and to show the substitution
                    }
                    ImplTraitId::TypeAliasImplTrait(alias, _) => alias.krate(db),
                };
                write_bounds_like_dyn_trait_with_prefix(
                    f,
                    "impl",
                    Either::Left(*self),
                    &bounds,
                    SizedByDefault::Sized { anchor: krate },
                )?;
            }
            TyKind::Closure(id, substs) => {
                let id = id.0;
                if f.display_kind.is_source_code() {
                    if !f.display_kind.allows_opaque() {
                        return Err(HirDisplayError::DisplaySourceCodeError(
                            DisplaySourceCodeError::OpaqueType,
                        ));
                    } else if f.closure_style != ClosureStyle::ImplFn {
                        never!("Only `impl Fn` is valid for displaying closures in source code");
                    }
                }
                match f.closure_style {
                    ClosureStyle::Hide => return write!(f, "{TYPE_HINT_TRUNCATION}"),
                    ClosureStyle::ClosureWithId => {
                        return write!(
                            f,
                            "{{closure#{:?}}}",
                            salsa::plumbing::AsId::as_id(&id).index()
                        );
                    }
                    ClosureStyle::ClosureWithSubst => {
                        write!(f, "{{closure#{:?}}}", salsa::plumbing::AsId::as_id(&id).index())?;
                        return hir_fmt_generics(f, substs.as_slice(), None, None);
                    }
                    _ => (),
                }
                let sig = substs
                    .split_closure_args_untupled()
                    .closure_sig_as_fn_ptr_ty
                    .callable_sig(interner);
                if let Some(sig) = sig {
                    let sig = sig.skip_binder();
                    let InternedClosure(def, _) = db.lookup_intern_closure(id);
                    let infer = InferenceResult::for_body(db, def);
                    let (_, kind) = infer.closure_info(id);
                    match f.closure_style {
                        ClosureStyle::ImplFn => write!(f, "impl {kind:?}(")?,
                        ClosureStyle::RANotation => write!(f, "|")?,
                        _ => unreachable!(),
                    }
                    if sig.inputs().is_empty() {
                    } else if f.should_truncate() {
                        write!(f, "{TYPE_HINT_TRUNCATION}")?;
                    } else {
                        f.write_joined(sig.inputs(), ", ")?;
                    };
                    match f.closure_style {
                        ClosureStyle::ImplFn => write!(f, ")")?,
                        ClosureStyle::RANotation => write!(f, "|")?,
                        _ => unreachable!(),
                    }
                    if f.closure_style == ClosureStyle::RANotation || !sig.output().is_unit() {
                        write!(f, " -> ")?;
                        sig.output().hir_fmt(f)?;
                    }
                } else {
                    write!(f, "{{closure}}")?;
                }
            }
            TyKind::CoroutineClosure(id, args) => {
                let id = id.0;
                if f.display_kind.is_source_code() {
                    if !f.display_kind.allows_opaque() {
                        return Err(HirDisplayError::DisplaySourceCodeError(
                            DisplaySourceCodeError::OpaqueType,
                        ));
                    } else if f.closure_style != ClosureStyle::ImplFn {
                        never!("Only `impl Fn` is valid for displaying closures in source code");
                    }
                }
                match f.closure_style {
                    ClosureStyle::Hide => return write!(f, "{TYPE_HINT_TRUNCATION}"),
                    ClosureStyle::ClosureWithId => {
                        return write!(
                            f,
                            "{{async closure#{:?}}}",
                            salsa::plumbing::AsId::as_id(&id).index()
                        );
                    }
                    ClosureStyle::ClosureWithSubst => {
                        write!(
                            f,
                            "{{async closure#{:?}}}",
                            salsa::plumbing::AsId::as_id(&id).index()
                        )?;
                        return hir_fmt_generics(f, args.as_slice(), None, None);
                    }
                    _ => (),
                }
                let CoroutineClosureArgsParts { closure_kind_ty, signature_parts_ty, .. } =
                    args.split_coroutine_closure_args();
                let kind = closure_kind_ty.to_opt_closure_kind().unwrap();
                let kind = match kind {
                    rustc_type_ir::ClosureKind::Fn => "AsyncFn",
                    rustc_type_ir::ClosureKind::FnMut => "AsyncFnMut",
                    rustc_type_ir::ClosureKind::FnOnce => "AsyncFnOnce",
                };
                let TyKind::FnPtr(coroutine_sig, _) = signature_parts_ty.kind() else {
                    unreachable!("invalid coroutine closure signature");
                };
                let coroutine_sig = coroutine_sig.skip_binder();
                let coroutine_inputs = coroutine_sig.inputs();
                let TyKind::Tuple(coroutine_inputs) = coroutine_inputs.as_slice()[1].kind() else {
                    unreachable!("invalid coroutine closure signature");
                };
                let TyKind::Tuple(coroutine_output) = coroutine_sig.output().kind() else {
                    unreachable!("invalid coroutine closure signature");
                };
                let coroutine_output = coroutine_output.as_slice()[1];
                match f.closure_style {
                    ClosureStyle::ImplFn => write!(f, "impl {kind}(")?,
                    ClosureStyle::RANotation => write!(f, "async |")?,
                    _ => unreachable!(),
                }
                if coroutine_inputs.is_empty() {
                } else if f.should_truncate() {
                    write!(f, "{TYPE_HINT_TRUNCATION}")?;
                } else {
                    f.write_joined(coroutine_inputs, ", ")?;
                };
                match f.closure_style {
                    ClosureStyle::ImplFn => write!(f, ")")?,
                    ClosureStyle::RANotation => write!(f, "|")?,
                    _ => unreachable!(),
                }
                if f.closure_style == ClosureStyle::RANotation || !coroutine_output.is_unit() {
                    write!(f, " -> ")?;
                    coroutine_output.hir_fmt(f)?;
                }
            }
            TyKind::Placeholder(_) => write!(f, "{{placeholder}}")?,
            TyKind::Param(param) => {
                // FIXME: We should not access `param.id`, it should be removed, and we should know the
                // parent from the formatted type.
                let generics = generics(db, param.id.parent());
                let param_data = &generics[param.id.local_id()];
                match param_data {
                    TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                        TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                            write!(
                                f,
                                "{}",
                                p.name
                                    .clone()
                                    .unwrap_or_else(Name::missing)
                                    .display(f.db, f.edition())
                            )?
                        }
                        TypeParamProvenance::ArgumentImplTrait => {
                            let bounds = GenericPredicates::query_all(f.db, param.id.parent())
                                .iter_identity_copied()
                                .filter(|wc| match wc.kind().skip_binder() {
                                    ClauseKind::Trait(tr) => tr.self_ty() == *self,
                                    ClauseKind::Projection(proj) => proj.self_ty() == *self,
                                    ClauseKind::TypeOutlives(to) => to.0 == *self,
                                    _ => false,
                                })
                                .collect::<Vec<_>>();
                            let krate = param.id.parent().module(db).krate(db);
                            write_bounds_like_dyn_trait_with_prefix(
                                f,
                                "impl",
                                Either::Left(*self),
                                &bounds,
                                SizedByDefault::Sized { anchor: krate },
                            )?;
                        }
                    },
                    TypeOrConstParamData::ConstParamData(p) => {
                        write!(f, "{}", p.name.display(f.db, f.edition()))?;
                    }
                }
            }
            TyKind::Bound(BoundVarIndexKind::Bound(debruijn), ty) => {
                write!(f, "?{}.{}", debruijn.as_usize(), ty.var.as_usize())?
            }
            TyKind::Bound(BoundVarIndexKind::Canonical, ty) => {
                write!(f, "?c.{}", ty.var.as_usize())?
            }
            TyKind::Dynamic(bounds, region) => {
                // We want to put auto traits after principal traits, regardless of their written order.
                let mut bounds_to_display = SmallVec::<[_; 4]>::new();
                let mut auto_trait_bounds = SmallVec::<[_; 4]>::new();
                for bound in bounds.iter() {
                    let clause = bound.with_self_ty(interner, *self);
                    match bound.skip_binder() {
                        ExistentialPredicate::Trait(_) | ExistentialPredicate::Projection(_) => {
                            bounds_to_display.push(clause);
                        }
                        ExistentialPredicate::AutoTrait(_) => auto_trait_bounds.push(clause),
                    }
                }
                bounds_to_display.append(&mut auto_trait_bounds);

                if f.render_region(region) {
                    bounds_to_display
                        .push(rustc_type_ir::OutlivesPredicate(*self, region).upcast(interner));
                }

                write_bounds_like_dyn_trait_with_prefix(
                    f,
                    "dyn",
                    Either::Left(*self),
                    &bounds_to_display,
                    SizedByDefault::NotSized,
                )?;
            }
            TyKind::Error(_) => {
                if f.display_kind.is_source_code() {
                    f.write_char('_')?;
                } else {
                    write!(f, "{{unknown}}")?;
                }
            }
            TyKind::Infer(..) => write!(f, "_")?,
            TyKind::Coroutine(coroutine_id, subst) => {
                let InternedCoroutine(owner, expr_id) = coroutine_id.0.loc(db);
                let CoroutineArgsParts { resume_ty, yield_ty, return_ty, .. } =
                    subst.split_coroutine_args();
                let body = db.body(owner);
                let expr = &body[expr_id];
                match expr {
                    hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::Async,
                        ..
                    }
                    | hir_def::hir::Expr::Async { .. } => {
                        let future_trait = f.lang_items().Future;
                        let output = future_trait.and_then(|t| {
                            t.trait_items(db)
                                .associated_type_by_name(&Name::new_symbol_root(sym::Output))
                        });
                        write!(f, "impl ")?;
                        if let Some(t) = future_trait {
                            f.start_location_link(t.into());
                        }
                        write!(f, "Future")?;
                        if future_trait.is_some() {
                            f.end_location_link();
                        }
                        write!(f, "<")?;
                        if let Some(t) = output {
                            f.start_location_link(t.into());
                        }
                        write!(f, "Output")?;
                        if output.is_some() {
                            f.end_location_link();
                        }
                        write!(f, " = ")?;
                        return_ty.hir_fmt(f)?;
                        write!(f, ">")?;
                    }
                    hir_def::hir::Expr::Closure {
                        closure_kind: hir_def::hir::ClosureKind::Coroutine(..),
                        ..
                    } => {
                        if f.display_kind.is_source_code() {
                            return Err(HirDisplayError::DisplaySourceCodeError(
                                DisplaySourceCodeError::Coroutine,
                            ));
                        }
                        write!(f, "|")?;
                        resume_ty.hir_fmt(f)?;
                        write!(f, "|")?;

                        write!(f, " yields ")?;
                        yield_ty.hir_fmt(f)?;

                        write!(f, " -> ")?;
                        return_ty.hir_fmt(f)?;
                    }
                    _ => panic!("invalid expr for coroutine: {expr:?}"),
                }
            }
            TyKind::CoroutineWitness(..) => write!(f, "{{coroutine witness}}")?,
            TyKind::Pat(_, _) => write!(f, "{{pat}}")?,
            TyKind::UnsafeBinder(_) => write!(f, "{{unsafe binder}}")?,
            TyKind::Alias(_, _) => write!(f, "{{alias}}")?,
        }
        Ok(())
    }
}

fn hir_fmt_generics<'db>(
    f: &mut HirFormatter<'_, 'db>,
    parameters: &[GenericArg<'db>],
    generic_def: Option<hir_def::GenericDefId>,
    self_: Option<Ty<'db>>,
) -> Result {
    if parameters.is_empty() {
        return Ok(());
    }

    let parameters_to_write = generic_args_sans_defaults(f, generic_def, parameters);

    if !parameters_to_write.is_empty() {
        write!(f, "<")?;
        hir_fmt_generic_arguments(f, parameters_to_write, self_)?;
        write!(f, ">")?;
    }

    Ok(())
}

fn generic_args_sans_defaults<'ga, 'db>(
    f: &mut HirFormatter<'_, 'db>,
    generic_def: Option<hir_def::GenericDefId>,
    parameters: &'ga [GenericArg<'db>],
) -> &'ga [GenericArg<'db>] {
    if f.display_kind.is_source_code() || f.omit_verbose_types() {
        match generic_def.map(|generic_def_id| f.db.generic_defaults(generic_def_id)) {
            None => parameters,
            Some(default_parameters) => {
                let should_show = |arg: GenericArg<'db>, i: usize| match default_parameters.get(i) {
                    None => true,
                    Some(default_parameter) => {
                        arg != default_parameter.instantiate(f.interner, &parameters[..i])
                    }
                };
                let mut default_from = 0;
                for (i, &parameter) in parameters.iter().enumerate() {
                    if should_show(parameter, i) {
                        default_from = i + 1;
                    }
                }
                &parameters[0..default_from]
            }
        }
    } else {
        parameters
    }
}

fn hir_fmt_generic_args<'db>(
    f: &mut HirFormatter<'_, 'db>,
    parameters: &[GenericArg<'db>],
    generic_def: Option<hir_def::GenericDefId>,
    self_: Option<Ty<'db>>,
) -> Result {
    if parameters.is_empty() {
        return Ok(());
    }

    let parameters_to_write = generic_args_sans_defaults(f, generic_def, parameters);

    if !parameters_to_write.is_empty() {
        write!(f, "<")?;
        hir_fmt_generic_arguments(f, parameters_to_write, self_)?;
        write!(f, ">")?;
    }

    Ok(())
}

fn hir_fmt_generic_arguments<'db>(
    f: &mut HirFormatter<'_, 'db>,
    parameters: &[GenericArg<'db>],
    self_: Option<Ty<'db>>,
) -> Result {
    let mut first = true;
    let lifetime_offset = parameters.iter().position(|arg| arg.region().is_some());

    let (ty_or_const, lifetimes) = match lifetime_offset {
        Some(offset) => parameters.split_at(offset),
        None => (parameters, &[][..]),
    };
    for generic_arg in lifetimes.iter().chain(ty_or_const) {
        if !mem::take(&mut first) {
            write!(f, ", ")?;
        }
        match self_ {
            self_ @ Some(_) if generic_arg.ty() == self_ => write!(f, "Self")?,
            _ => generic_arg.hir_fmt(f)?,
        }
    }
    Ok(())
}

fn hir_fmt_tys<'db>(
    f: &mut HirFormatter<'_, 'db>,
    tys: &[Ty<'db>],
    self_: Option<Ty<'db>>,
) -> Result {
    let mut first = true;

    for ty in tys {
        if !mem::take(&mut first) {
            write!(f, ", ")?;
        }
        match self_ {
            Some(self_) if *ty == self_ => write!(f, "Self")?,
            _ => ty.hir_fmt(f)?,
        }
    }
    Ok(())
}

impl<'db> HirDisplay<'db> for PolyFnSig<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        let FnSig { inputs_and_output, c_variadic, safety, abi: _ } = self.skip_binder();
        if let Safety::Unsafe = safety {
            write!(f, "unsafe ")?;
        }
        // FIXME: Enable this when the FIXME on FnAbi regarding PartialEq is fixed.
        // if !matches!(abi, FnAbi::Rust) {
        //     f.write_str("extern \"")?;
        //     f.write_str(abi.as_str())?;
        //     f.write_str("\" ")?;
        // }
        write!(f, "fn(")?;
        f.write_joined(inputs_and_output.inputs(), ", ")?;
        if c_variadic {
            if inputs_and_output.inputs().is_empty() {
                write!(f, "...")?;
            } else {
                write!(f, ", ...")?;
            }
        }
        write!(f, ")")?;
        let ret = inputs_and_output.output();
        if !ret.is_unit() {
            write!(f, " -> ")?;
            ret.hir_fmt(f)?;
        }
        Ok(())
    }
}

impl<'db> HirDisplay<'db> for Term<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        match self {
            Term::Ty(it) => it.hir_fmt(f),
            Term::Const(it) => it.hir_fmt(f),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum SizedByDefault {
    NotSized,
    Sized { anchor: Crate },
}

impl SizedByDefault {
    fn is_sized_trait(self, trait_: TraitId, db: &dyn DefDatabase) -> bool {
        match self {
            Self::NotSized => false,
            Self::Sized { anchor } => {
                let sized_trait = hir_def::lang_item::lang_items(db, anchor).Sized;
                Some(trait_) == sized_trait
            }
        }
    }
}

pub fn write_bounds_like_dyn_trait_with_prefix<'db>(
    f: &mut HirFormatter<'_, 'db>,
    prefix: &str,
    this: Either<Ty<'db>, Region<'db>>,
    predicates: &[Clause<'db>],
    default_sized: SizedByDefault,
) -> Result {
    write!(f, "{prefix}")?;
    if !predicates.is_empty()
        || predicates.is_empty() && matches!(default_sized, SizedByDefault::Sized { .. })
    {
        write!(f, " ")?;
        write_bounds_like_dyn_trait(f, this, predicates, default_sized)
    } else {
        Ok(())
    }
}

fn write_bounds_like_dyn_trait<'db>(
    f: &mut HirFormatter<'_, 'db>,
    this: Either<Ty<'db>, Region<'db>>,
    predicates: &[Clause<'db>],
    default_sized: SizedByDefault,
) -> Result {
    // Note: This code is written to produce nice results (i.e.
    // corresponding to surface Rust) for types that can occur in
    // actual Rust. It will have weird results if the predicates
    // aren't as expected (i.e. self types = $0, projection
    // predicates for a certain trait come after the Implemented
    // predicate for that trait).
    let mut first = true;
    let mut angle_open = false;
    let mut is_fn_trait = false;
    let mut is_sized = false;
    for p in predicates {
        match p.kind().skip_binder() {
            ClauseKind::Trait(trait_ref) => {
                let trait_ = trait_ref.def_id().0;
                if default_sized.is_sized_trait(trait_, f.db) {
                    is_sized = true;
                    if matches!(default_sized, SizedByDefault::Sized { .. }) {
                        // Don't print +Sized, but rather +?Sized if absent.
                        continue;
                    }
                }
                if !is_fn_trait {
                    is_fn_trait = fn_traits(f.lang_items()).any(|it| it == trait_);
                }
                if !is_fn_trait && angle_open {
                    write!(f, ">")?;
                    angle_open = false;
                }
                if !first {
                    write!(f, " + ")?;
                }
                // We assume that the self type is ^0.0 (i.e. the
                // existential) here, which is the only thing that's
                // possible in actual Rust, and hence don't print it
                f.start_location_link(trait_.into());
                write!(f, "{}", f.db.trait_signature(trait_).name.display(f.db, f.edition()))?;
                f.end_location_link();
                if is_fn_trait {
                    if let [_self, params @ ..] = trait_ref.trait_ref.args.as_slice()
                        && let Some(args) = params.first().and_then(|it| it.ty()?.as_tuple())
                    {
                        write!(f, "(")?;
                        hir_fmt_tys(f, args.as_slice(), Some(trait_ref.trait_ref.self_ty()))?;
                        write!(f, ")")?;
                    }
                } else {
                    let params = generic_args_sans_defaults(
                        f,
                        Some(trait_.into()),
                        trait_ref.trait_ref.args.as_slice(),
                    );
                    if let [_self, params @ ..] = params
                        && !params.is_empty()
                    {
                        write!(f, "<")?;
                        hir_fmt_generic_arguments(f, params, Some(trait_ref.trait_ref.self_ty()))?;
                        // there might be assoc type bindings, so we leave the angle brackets open
                        angle_open = true;
                    }
                }
            }
            ClauseKind::TypeOutlives(to) if Either::Left(to.0) == this => {
                if !is_fn_trait && angle_open {
                    write!(f, ">")?;
                    angle_open = false;
                }
                if !first {
                    write!(f, " + ")?;
                }
                to.1.hir_fmt(f)?;
            }
            ClauseKind::RegionOutlives(lo) if Either::Right(lo.0) == this => {
                if !is_fn_trait && angle_open {
                    write!(f, ">")?;
                    angle_open = false;
                }
                if !first {
                    write!(f, " + ")?;
                }
                lo.1.hir_fmt(f)?;
            }
            ClauseKind::Projection(projection) if is_fn_trait => {
                is_fn_trait = false;
                if !projection.term.as_type().is_some_and(|it| it.is_unit()) {
                    write!(f, " -> ")?;
                    projection.term.hir_fmt(f)?;
                }
            }
            ClauseKind::Projection(projection) => {
                // in types in actual Rust, these will always come
                // after the corresponding Implemented predicate
                if angle_open {
                    write!(f, ", ")?;
                } else {
                    write!(f, "<")?;
                    angle_open = true;
                }
                let assoc_ty_id = projection.def_id().expect_type_alias();
                let type_alias = f.db.type_alias_signature(assoc_ty_id);
                f.start_location_link(assoc_ty_id.into());
                write!(f, "{}", type_alias.name.display(f.db, f.edition()))?;
                f.end_location_link();

                let own_args = projection.projection_term.own_args(f.interner);
                if !own_args.is_empty() {
                    write!(f, "<")?;
                    hir_fmt_generic_arguments(f, own_args.as_slice(), None)?;
                    write!(f, ">")?;
                }
                write!(f, " = ")?;
                projection.term.hir_fmt(f)?;
            }
            _ => {}
        }
        first = false;
    }
    if angle_open {
        write!(f, ">")?;
    }
    if let SizedByDefault::Sized { anchor } = default_sized {
        let sized_trait = hir_def::lang_item::lang_items(f.db, anchor).Sized;
        if !is_sized {
            if !first {
                write!(f, " + ")?;
            }
            if let Some(sized_trait) = sized_trait {
                f.start_location_link(sized_trait.into());
            }
            write!(f, "?Sized")?;
        } else if first {
            if let Some(sized_trait) = sized_trait {
                f.start_location_link(sized_trait.into());
            }
            write!(f, "Sized")?;
        }
        if sized_trait.is_some() {
            f.end_location_link();
        }
    }
    Ok(())
}

impl<'db> HirDisplay<'db> for TraitRef<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        let trait_ = self.def_id.0;
        f.start_location_link(trait_.into());
        write!(f, "{}", f.db.trait_signature(trait_).name.display(f.db, f.edition()))?;
        f.end_location_link();
        let substs = self.args.as_slice();
        hir_fmt_generic_args(f, &substs[1..], None, Some(self.self_ty()))
    }
}

impl<'db> HirDisplay<'db> for Region<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        match self.kind() {
            RegionKind::ReEarlyParam(param) => {
                let generics = generics(f.db, param.id.parent);
                let param_data = &generics[param.id.local_id];
                write!(f, "{}", param_data.name.display(f.db, f.edition()))?;
                Ok(())
            }
            RegionKind::ReBound(BoundVarIndexKind::Bound(db), idx) => {
                write!(f, "?{}.{}", db.as_u32(), idx.var.as_u32())
            }
            RegionKind::ReBound(BoundVarIndexKind::Canonical, idx) => {
                write!(f, "?c.{}", idx.var.as_u32())
            }
            RegionKind::ReVar(_) => write!(f, "_"),
            RegionKind::ReStatic => write!(f, "'static"),
            RegionKind::ReError(..) => {
                if cfg!(test) {
                    write!(f, "'?")
                } else {
                    write!(f, "'_")
                }
            }
            RegionKind::ReErased => write!(f, "'<erased>"),
            RegionKind::RePlaceholder(_) => write!(f, "<placeholder>"),
            RegionKind::ReLateParam(_) => write!(f, "<late-param>"),
        }
    }
}

pub fn write_visibility<'db>(
    module_id: ModuleId,
    vis: Visibility,
    f: &mut HirFormatter<'_, 'db>,
) -> Result {
    match vis {
        Visibility::Public => write!(f, "pub "),
        Visibility::PubCrate(_) => write!(f, "pub(crate) "),
        Visibility::Module(vis_id, _) => {
            let def_map = module_id.def_map(f.db);
            let root_module_id = def_map.root_module_id();
            if vis_id == module_id {
                // pub(self) or omitted
                Ok(())
            } else if root_module_id == vis_id && root_module_id.block(f.db).is_none() {
                write!(f, "pub(crate) ")
            } else if module_id.containing_module(f.db) == Some(vis_id)
                && !vis_id.is_block_module(f.db)
            {
                write!(f, "pub(super) ")
            } else {
                write!(f, "pub(in ...) ")
            }
        }
    }
}

pub trait HirDisplayWithExpressionStore<'db> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result;
}

impl<'db, T: ?Sized + HirDisplayWithExpressionStore<'db>> HirDisplayWithExpressionStore<'db>
    for &'_ T
{
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        T::hir_fmt(&**self, f, store)
    }
}

pub fn hir_display_with_store<'a, 'db, T: HirDisplayWithExpressionStore<'db> + 'a>(
    value: T,
    store: &'a ExpressionStore,
) -> impl HirDisplay<'db> + 'a {
    ExpressionStoreAdapter(value, store)
}

struct ExpressionStoreAdapter<'a, T>(T, &'a ExpressionStore);

impl<'a, T> ExpressionStoreAdapter<'a, T> {
    fn wrap(store: &'a ExpressionStore) -> impl Fn(T) -> ExpressionStoreAdapter<'a, T> {
        move |value| ExpressionStoreAdapter(value, store)
    }
}

impl<'db, T: HirDisplayWithExpressionStore<'db>> HirDisplay<'db> for ExpressionStoreAdapter<'_, T> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>) -> Result {
        T::hir_fmt(&self.0, f, self.1)
    }
}
impl<'db> HirDisplayWithExpressionStore<'db> for LifetimeRefId {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        match &store[*self] {
            LifetimeRef::Named(name) => write!(f, "{}", name.display(f.db, f.edition())),
            LifetimeRef::Static => write!(f, "'static"),
            LifetimeRef::Placeholder => write!(f, "'_"),
            LifetimeRef::Error => write!(f, "'{{error}}"),
            &LifetimeRef::Param(lifetime_param_id) => {
                let generic_params = f.db.generic_params(lifetime_param_id.parent);
                write!(
                    f,
                    "{}",
                    generic_params[lifetime_param_id.local_id].name.display(f.db, f.edition())
                )
            }
        }
    }
}

impl<'db> HirDisplayWithExpressionStore<'db> for TypeRefId {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        match &store[*self] {
            TypeRef::Never => write!(f, "!")?,
            TypeRef::TypeParam(param) => {
                let generic_params = f.db.generic_params(param.parent());
                match generic_params[param.local_id()].name() {
                    Some(name) => write!(f, "{}", name.display(f.db, f.edition()))?,
                    None => {
                        write!(f, "impl ")?;
                        f.write_joined(
                            generic_params
                                .where_predicates()
                                .iter()
                                .filter_map(|it| match it {
                                    WherePredicate::TypeBound { target, bound }
                                    | WherePredicate::ForLifetime { lifetimes: _, target, bound }
                                        if matches!(
                                            store[*target],
                                            TypeRef::TypeParam(t) if t == *param
                                        ) =>
                                    {
                                        Some(bound)
                                    }
                                    _ => None,
                                })
                                .map(ExpressionStoreAdapter::wrap(store)),
                            " + ",
                        )?;
                    }
                }
            }
            TypeRef::Placeholder => write!(f, "_")?,
            TypeRef::Tuple(elems) => {
                write!(f, "(")?;
                f.write_joined(elems.iter().map(ExpressionStoreAdapter::wrap(store)), ", ")?;
                if elems.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")?;
            }
            TypeRef::Path(path) => path.hir_fmt(f, store)?,
            TypeRef::RawPtr(inner, mutability) => {
                let mutability = match mutability {
                    hir_def::type_ref::Mutability::Shared => "*const ",
                    hir_def::type_ref::Mutability::Mut => "*mut ",
                };
                write!(f, "{mutability}")?;
                inner.hir_fmt(f, store)?;
            }
            TypeRef::Reference(ref_) => {
                let mutability = match ref_.mutability {
                    hir_def::type_ref::Mutability::Shared => "",
                    hir_def::type_ref::Mutability::Mut => "mut ",
                };
                write!(f, "&")?;
                if let Some(lifetime) = &ref_.lifetime {
                    lifetime.hir_fmt(f, store)?;
                    write!(f, " ")?;
                }
                write!(f, "{mutability}")?;
                ref_.ty.hir_fmt(f, store)?;
            }
            TypeRef::Array(array) => {
                write!(f, "[")?;
                array.ty.hir_fmt(f, store)?;
                write!(f, "; ")?;
                array.len.hir_fmt(f, store)?;
                write!(f, "]")?;
            }
            TypeRef::Slice(inner) => {
                write!(f, "[")?;
                inner.hir_fmt(f, store)?;
                write!(f, "]")?;
            }
            TypeRef::Fn(fn_) => {
                if fn_.is_unsafe {
                    write!(f, "unsafe ")?;
                }
                if let Some(abi) = &fn_.abi {
                    f.write_str("extern \"")?;
                    f.write_str(abi.as_str())?;
                    f.write_str("\" ")?;
                }
                write!(f, "fn(")?;
                if let Some(((_, return_type), function_parameters)) = fn_.params.split_last() {
                    for index in 0..function_parameters.len() {
                        let (param_name, param_type) = &function_parameters[index];
                        if let Some(name) = param_name {
                            write!(f, "{}: ", name.display(f.db, f.edition()))?;
                        }

                        param_type.hir_fmt(f, store)?;

                        if index != function_parameters.len() - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if fn_.is_varargs {
                        write!(f, "{}...", if fn_.params.len() == 1 { "" } else { ", " })?;
                    }
                    write!(f, ")")?;
                    match &store[*return_type] {
                        TypeRef::Tuple(tup) if tup.is_empty() => {}
                        _ => {
                            write!(f, " -> ")?;
                            return_type.hir_fmt(f, store)?;
                        }
                    }
                }
            }
            TypeRef::ImplTrait(bounds) => {
                write!(f, "impl ")?;
                f.write_joined(bounds.iter().map(ExpressionStoreAdapter::wrap(store)), " + ")?;
            }
            TypeRef::DynTrait(bounds) => {
                write!(f, "dyn ")?;
                f.write_joined(bounds.iter().map(ExpressionStoreAdapter::wrap(store)), " + ")?;
            }
            TypeRef::Error => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl<'db> HirDisplayWithExpressionStore<'db> for ConstRef {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, _store: &ExpressionStore) -> Result {
        // FIXME
        write!(f, "{{const}}")?;

        Ok(())
    }
}

impl<'db> HirDisplayWithExpressionStore<'db> for TypeBound {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        match self {
            &TypeBound::Path(path, modifier) => {
                match modifier {
                    TraitBoundModifier::None => (),
                    TraitBoundModifier::Maybe => write!(f, "?")?,
                }
                store[path].hir_fmt(f, store)
            }
            TypeBound::Lifetime(lifetime) => lifetime.hir_fmt(f, store),
            TypeBound::ForLifetime(lifetimes, path) => {
                let edition = f.edition();
                write!(
                    f,
                    "for<{}> ",
                    lifetimes.iter().map(|it| it.display(f.db, edition)).format(", ")
                )?;
                store[*path].hir_fmt(f, store)
            }
            TypeBound::Use(args) => {
                write!(f, "use<")?;
                let edition = f.edition();
                let last = args.len().saturating_sub(1);
                for (idx, arg) in args.iter().enumerate() {
                    match arg {
                        UseArgRef::Lifetime(lt) => lt.hir_fmt(f, store)?,
                        UseArgRef::Name(n) => write!(f, "{}", n.display(f.db, edition))?,
                    }
                    if idx != last {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "> ")
            }
            TypeBound::Error => write!(f, "{{error}}"),
        }
    }
}

impl<'db> HirDisplayWithExpressionStore<'db> for Path {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        match (self.type_anchor(), self.kind()) {
            (Some(anchor), _) => {
                write!(f, "<")?;
                anchor.hir_fmt(f, store)?;
                write!(f, ">")?;
            }
            (_, PathKind::Plain) => {}
            (_, PathKind::Abs) => {}
            (_, PathKind::Crate) => write!(f, "crate")?,
            (_, &PathKind::SELF) => write!(f, "self")?,
            (_, PathKind::Super(n)) => {
                for i in 0..*n {
                    if i > 0 {
                        write!(f, "::")?;
                    }
                    write!(f, "super")?;
                }
            }
            (_, PathKind::DollarCrate(id)) => {
                // Resolve `$crate` to the crate's display name.
                // FIXME: should use the dependency name instead if available, but that depends on
                // the crate invoking `HirDisplay`
                let crate_data = id.extra_data(f.db);
                let name = crate_data
                    .display_name
                    .as_ref()
                    .map(|name| (*name.canonical_name()).clone())
                    .unwrap_or(sym::dollar_crate);
                write!(f, "{name}")?
            }
        }

        // Convert trait's `Self` bound back to the surface syntax. Note there is no associated
        // trait, so there can only be one path segment that `has_self_type`. The `Self` type
        // itself can contain further qualified path through, which will be handled by recursive
        // `hir_fmt`s.
        //
        // `trait_mod::Trait<Self = type_mod::Type, Args>::Assoc`
        // =>
        // `<type_mod::Type as trait_mod::Trait<Args>>::Assoc`
        let trait_self_ty = self.segments().iter().find_map(|seg| {
            let generic_args = seg.args_and_bindings?;
            generic_args.has_self_type.then(|| &generic_args.args[0])
        });
        if let Some(ty) = trait_self_ty {
            write!(f, "<")?;
            ty.hir_fmt(f, store)?;
            write!(f, " as ")?;
            // Now format the path of the trait...
        }

        for (seg_idx, segment) in self.segments().iter().enumerate() {
            if !matches!(self.kind(), PathKind::Plain) || seg_idx > 0 {
                write!(f, "::")?;
            }
            write!(f, "{}", segment.name.display(f.db, f.edition()))?;
            if let Some(generic_args) = segment.args_and_bindings {
                // We should be in type context, so format as `Foo<Bar>` instead of `Foo::<Bar>`.
                // Do we actually format expressions?
                match generic_args.parenthesized {
                    hir_def::expr_store::path::GenericArgsParentheses::ReturnTypeNotation => {
                        write!(f, "(..)")?;
                    }
                    hir_def::expr_store::path::GenericArgsParentheses::ParenSugar => {
                        // First argument will be a tuple, which already includes the parentheses.
                        // If the tuple only contains 1 item, write it manually to avoid the trailing `,`.
                        let tuple = match generic_args.args[0] {
                            hir_def::expr_store::path::GenericArg::Type(ty) => match &store[ty] {
                                TypeRef::Tuple(it) => Some(it),
                                _ => None,
                            },
                            _ => None,
                        };
                        if let Some(v) = tuple {
                            if v.len() == 1 {
                                write!(f, "(")?;
                                v[0].hir_fmt(f, store)?;
                                write!(f, ")")?;
                            } else {
                                generic_args.args[0].hir_fmt(f, store)?;
                            }
                        }
                        if let Some(ret) = generic_args.bindings[0].type_ref
                            && !matches!(&store[ret], TypeRef::Tuple(v) if v.is_empty())
                        {
                            write!(f, " -> ")?;
                            ret.hir_fmt(f, store)?;
                        }
                    }
                    hir_def::expr_store::path::GenericArgsParentheses::No => {
                        let mut first = true;
                        // Skip the `Self` bound if exists. It's handled outside the loop.
                        for arg in &generic_args.args[generic_args.has_self_type as usize..] {
                            if first {
                                first = false;
                                write!(f, "<")?;
                            } else {
                                write!(f, ", ")?;
                            }
                            arg.hir_fmt(f, store)?;
                        }
                        for binding in generic_args.bindings.iter() {
                            if first {
                                first = false;
                                write!(f, "<")?;
                            } else {
                                write!(f, ", ")?;
                            }
                            write!(f, "{}", binding.name.display(f.db, f.edition()))?;
                            match &binding.type_ref {
                                Some(ty) => {
                                    write!(f, " = ")?;
                                    ty.hir_fmt(f, store)?
                                }
                                None => {
                                    write!(f, ": ")?;
                                    f.write_joined(
                                        binding
                                            .bounds
                                            .iter()
                                            .map(ExpressionStoreAdapter::wrap(store)),
                                        " + ",
                                    )?;
                                }
                            }
                        }

                        // There may be no generic arguments to print, in case of a trait having only a
                        // single `Self` bound which is converted to `<Ty as Trait>::Assoc`.
                        if !first {
                            write!(f, ">")?;
                        }

                        // Current position: `<Ty as Trait<Args>|`
                        if generic_args.has_self_type {
                            write!(f, ">")?;
                        }
                    }
                }
            }
        }

        Ok(())
    }
}

impl<'db> HirDisplayWithExpressionStore<'db> for hir_def::expr_store::path::GenericArg {
    fn hir_fmt(&self, f: &mut HirFormatter<'_, 'db>, store: &ExpressionStore) -> Result {
        match self {
            hir_def::expr_store::path::GenericArg::Type(ty) => ty.hir_fmt(f, store),
            hir_def::expr_store::path::GenericArg::Const(_c) => {
                // write!(f, "{}", c.display(f.db, f.edition()))
                write!(f, "<expr>")
            }
            hir_def::expr_store::path::GenericArg::Lifetime(lifetime) => lifetime.hir_fmt(f, store),
        }
    }
}

//! The `HirDisplay` trait, which serves two purposes: Turning various bits from
//! HIR back into source code, and just displaying them for debugging/testing
//! purposes.

use std::{
    fmt::{self, Debug},
    mem,
};

use base_db::Crate;
use chalk_ir::{BoundVar, Safety, TyKind};
use either::Either;
use hir_def::{
    GenericDefId, HasModule, ImportPathConfig, ItemContainerId, LocalFieldId, Lookup, ModuleDefId,
    ModuleId, TraitId,
    db::DefDatabase,
    expr_store::{ExpressionStore, path::Path},
    find_path::{self, PrefixKind},
    hir::generics::{TypeOrConstParamData, TypeParamProvenance, WherePredicate},
    item_scope::ItemInNs,
    item_tree::FieldsShape,
    lang_item::LangItem,
    nameres::DefMap,
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
use rustc_hash::FxHashSet;
use smallvec::SmallVec;
use span::Edition;
use stdx::never;
use triomphe::Arc;

use crate::{
    AdtId, AliasEq, AliasTy, Binders, CallableDefId, CallableSig, ConcreteConst, Const,
    ConstScalar, ConstValue, DomainGoal, FnAbi, GenericArg, ImplTraitId, Interner, Lifetime,
    LifetimeData, LifetimeOutlives, MemoryMap, Mutability, OpaqueTy, ProjectionTy, ProjectionTyExt,
    QuantifiedWhereClause, Scalar, Substitution, TraitEnvironment, TraitRef, TraitRefExt, Ty,
    TyExt, WhereClause,
    consteval::try_const_usize,
    db::{HirDatabase, InternedClosure},
    from_assoc_type_id, from_foreign_def_id, from_placeholder_idx,
    generics::generics,
    infer::normalize,
    layout::Layout,
    lt_from_placeholder_idx,
    mapping::from_chalk,
    mir::pad16,
    primitive, to_assoc_type_id,
    utils::{self, ClosureSubst, detect_variant_from_bytes},
};

pub trait HirWrite: fmt::Write {
    fn start_location_link(&mut self, _location: ModuleDefId) {}
    fn end_location_link(&mut self) {}
}

// String will ignore link metadata
impl HirWrite for String {}

// `core::Formatter` will ignore metadata
impl HirWrite for fmt::Formatter<'_> {}

pub struct HirFormatter<'a> {
    /// The database handle
    pub db: &'a dyn HirDatabase,
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
    bounds_formatting_ctx: BoundsFormattingCtx,
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
enum BoundsFormattingCtx {
    Entered {
        /// We can have recursive bounds like the following case:
        /// ```ignore
        /// where
        ///     T: Foo,
        ///     T::FooAssoc: Baz<<T::FooAssoc as Bar>::BarAssoc> + Bar
        /// ```
        /// So, record the projection types met while formatting bounds and
        //. prevent recursing into their bounds to avoid infinite loops.
        projection_tys_met: FxHashSet<ProjectionTy>,
    },
    #[default]
    Exited,
}

impl BoundsFormattingCtx {
    fn contains(&mut self, proj: &ProjectionTy) -> bool {
        match self {
            BoundsFormattingCtx::Entered { projection_tys_met } => {
                projection_tys_met.contains(proj)
            }
            BoundsFormattingCtx::Exited => false,
        }
    }
}

impl HirFormatter<'_> {
    fn start_location_link(&mut self, location: ModuleDefId) {
        self.fmt.start_location_link(location);
    }

    fn end_location_link(&mut self) {
        self.fmt.end_location_link();
    }

    fn format_bounds_with<T, F: FnOnce(&mut Self) -> T>(
        &mut self,
        target: ProjectionTy,
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

    fn render_lifetime(&self, lifetime: &Lifetime) -> bool {
        match self.display_lifetimes {
            DisplayLifetime::Always => true,
            DisplayLifetime::OnlyStatic => matches!(***lifetime.interned(), LifetimeData::Static),
            DisplayLifetime::OnlyNamed => {
                matches!(***lifetime.interned(), LifetimeData::Placeholder(_))
            }
            DisplayLifetime::OnlyNamedOrStatic => matches!(
                ***lifetime.interned(),
                LifetimeData::Static | LifetimeData::Placeholder(_)
            ),
            DisplayLifetime::Never => false,
        }
    }
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError>;

    /// Returns a `Display`able type that is human-readable.
    fn into_displayable<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        max_size: Option<usize>,
        limited_size: Option<usize>,
        omit_verbose_types: bool,
        display_target: DisplayTarget,
        display_kind: DisplayKind,
        closure_style: ClosureStyle,
        show_container_bounds: bool,
    ) -> HirDisplayWrapper<'a, Self>
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
        db: &'a dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
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
        db: &'a dyn HirDatabase,
        max_size: Option<usize>,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
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
        db: &'a dyn HirDatabase,
        limited_size: Option<usize>,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
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
        db: &'a dyn HirDatabase,
        module_id: ModuleId,
        allow_opaque: bool,
    ) -> Result<String, DisplaySourceCodeError> {
        let mut result = String::new();
        match self.hir_fmt(&mut HirFormatter {
            db,
            fmt: &mut result,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: None,
            entity_limit: None,
            omit_verbose_types: false,
            closure_style: ClosureStyle::ImplFn,
            display_target: DisplayTarget::from_crate(db, module_id.krate()),
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
        db: &'a dyn HirDatabase,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
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
        db: &'a dyn HirDatabase,
        show_container_bounds: bool,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
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

impl HirFormatter<'_> {
    pub fn krate(&self) -> Crate {
        self.display_target.krate
    }

    pub fn edition(&self) -> Edition {
        self.display_target.edition
    }

    pub fn write_joined<T: HirDisplay>(
        &mut self,
        iter: impl IntoIterator<Item = T>,
        sep: &str,
    ) -> Result<(), HirDisplayError> {
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
    pub fn write_fmt(&mut self, args: fmt::Arguments<'_>) -> Result<(), HirDisplayError> {
        // We write to a buffer first to track output size
        self.buf.clear();
        fmt::write(&mut self.buf, args)?;
        self.curr_size += self.buf.len();

        // Then we write to the internal formatter from the buffer
        self.fmt.write_str(&self.buf).map_err(HirDisplayError::from)
    }

    pub fn write_str(&mut self, s: &str) -> Result<(), HirDisplayError> {
        self.fmt.write_str(s)?;
        Ok(())
    }

    pub fn write_char(&mut self, c: char) -> Result<(), HirDisplayError> {
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

    fn is_test(self) -> bool {
        matches!(self, Self::Test)
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

pub struct HirDisplayWrapper<'a, T> {
    db: &'a dyn HirDatabase,
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

impl<T: HirDisplay> HirDisplayWrapper<'_, T> {
    pub fn write_to<F: HirWrite>(&self, f: &mut F) -> Result<(), HirDisplayError> {
        self.t.hir_fmt(&mut HirFormatter {
            db: self.db,
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

impl<T> fmt::Display for HirDisplayWrapper<'_, T>
where
    T: HirDisplay,
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

impl<T: HirDisplay> HirDisplay for &T {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl<T: HirDisplay + Internable> HirDisplay for Interned<T> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        HirDisplay::hir_fmt(self.as_ref(), f)
    }
}

impl HirDisplay for ProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }
        let trait_ref = self.trait_ref(f.db);
        let self_ty = trait_ref.self_type_parameter(Interner);

        // if we are projection on a type parameter, check if the projection target has bounds
        // itself, if so, we render them directly as `impl Bound` instead of the less useful
        // `<Param as Trait>::Assoc`
        if !f.display_kind.is_source_code() {
            if let TyKind::Placeholder(idx) = self_ty.kind(Interner) {
                if !f.bounds_formatting_ctx.contains(self) {
                    let db = f.db;
                    let id = from_placeholder_idx(db, *idx);
                    let generics = generics(db, id.parent);

                    let substs = generics.placeholder_subst(db);
                    let bounds = db
                        .generic_predicates(id.parent)
                        .iter()
                        .map(|pred| pred.clone().substitute(Interner, &substs))
                        .filter(|wc| match wc.skip_binders() {
                            WhereClause::Implemented(tr) => {
                                matches!(
                                    tr.self_type_parameter(Interner).kind(Interner),
                                    TyKind::Alias(_)
                                )
                            }
                            WhereClause::TypeOutlives(t) => {
                                matches!(t.ty.kind(Interner), TyKind::Alias(_))
                            }
                            // We shouldn't be here if these exist
                            WhereClause::AliasEq(_) => false,
                            WhereClause::LifetimeOutlives(_) => false,
                        })
                        .collect::<Vec<_>>();
                    if !bounds.is_empty() {
                        return f.format_bounds_with(self.clone(), |f| {
                            write_bounds_like_dyn_trait_with_prefix(
                                f,
                                "impl",
                                Either::Left(
                                    &TyKind::Alias(AliasTy::Projection(self.clone()))
                                        .intern(Interner),
                                ),
                                &bounds,
                                SizedByDefault::NotSized,
                            )
                        });
                    }
                }
            }
        }

        write!(f, "<")?;
        self_ty.hir_fmt(f)?;
        write!(f, " as ")?;
        trait_ref.hir_fmt(f)?;
        write!(
            f,
            ">::{}",
            f.db.type_alias_signature(from_assoc_type_id(self.associated_ty_id))
                .name
                .display(f.db, f.edition())
        )?;
        let proj_params =
            &self.substitution.as_slice(Interner)[trait_ref.substitution.len(Interner)..];
        hir_fmt_generics(f, proj_params, None, None)
    }
}

impl HirDisplay for OpaqueTy {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }

        self.substitution.at(Interner, 0).hir_fmt(f)
    }
}

impl HirDisplay for GenericArg {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self.interned() {
            crate::GenericArgData::Ty(ty) => ty.hir_fmt(f),
            crate::GenericArgData::Lifetime(lt) => lt.hir_fmt(f),
            crate::GenericArgData::Const(c) => c.hir_fmt(f),
        }
    }
}

impl HirDisplay for Const {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let data = self.interned();
        match &data.value {
            ConstValue::BoundVar(idx) => idx.hir_fmt(f),
            ConstValue::InferenceVar(..) => write!(f, "#c#"),
            ConstValue::Placeholder(idx) => {
                let id = from_placeholder_idx(f.db, *idx);
                let generics = generics(f.db, id.parent);
                let param_data = &generics[id.local_id];
                write!(f, "{}", param_data.name().unwrap().display(f.db, f.edition()))?;
                Ok(())
            }
            ConstValue::Concrete(c) => match &c.interned {
                ConstScalar::Bytes(b, m) => render_const_scalar(f, b, m, &data.ty),
                ConstScalar::UnevaluatedConst(c, parameters) => {
                    write!(f, "{}", c.name(f.db))?;
                    hir_fmt_generics(f, parameters.as_slice(Interner), c.generic_def(f.db), None)?;
                    Ok(())
                }
                ConstScalar::Unknown => f.write_char('_'),
            },
        }
    }
}

fn render_const_scalar(
    f: &mut HirFormatter<'_>,
    b: &[u8],
    memory_map: &MemoryMap,
    ty: &Ty,
) -> Result<(), HirDisplayError> {
    let trait_env = TraitEnvironment::empty(f.krate());
    let ty = normalize(f.db, trait_env.clone(), ty.clone());
    match ty.kind(Interner) {
        TyKind::Scalar(s) => match s {
            Scalar::Bool => write!(f, "{}", b[0] != 0),
            Scalar::Char => {
                let it = u128::from_le_bytes(pad16(b, false)) as u32;
                let Ok(c) = char::try_from(it) else {
                    return f.write_str("<unicode-error>");
                };
                write!(f, "{c:?}")
            }
            Scalar::Int(_) => {
                let it = i128::from_le_bytes(pad16(b, true));
                write!(f, "{it}")
            }
            Scalar::Uint(_) => {
                let it = u128::from_le_bytes(pad16(b, false));
                write!(f, "{it}")
            }
            Scalar::Float(fl) => match fl {
                chalk_ir::FloatTy::F16 => {
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
                chalk_ir::FloatTy::F32 => {
                    let it = f32::from_le_bytes(b.try_into().unwrap());
                    write!(f, "{it:?}")
                }
                chalk_ir::FloatTy::F64 => {
                    let it = f64::from_le_bytes(b.try_into().unwrap());
                    write!(f, "{it:?}")
                }
                chalk_ir::FloatTy::F128 => {
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
        },
        TyKind::Ref(_, _, t) => match t.kind(Interner) {
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
                let Ok(layout) = f.db.layout_of_ty(ty.clone(), trait_env) else {
                    return f.write_str("<layout-error>");
                };
                let size_one = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size_one * count) else {
                    return f.write_str("<ref-data-not-available>");
                };
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
            TyKind::Dyn(_) => {
                let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
                let ty_id = usize::from_le_bytes(b[b.len() / 2..].try_into().unwrap());
                let Ok(t) = memory_map.vtable_ty(ty_id) else {
                    return f.write_str("<ty-missing-in-vtable-map>");
                };
                let Ok(layout) = f.db.layout_of_ty(t.clone(), trait_env) else {
                    return f.write_str("<layout-error>");
                };
                let size = layout.size.bytes_usize();
                let Some(bytes) = memory_map.get(addr, size) else {
                    return f.write_str("<ref-data-not-available>");
                };
                f.write_str("&")?;
                render_const_scalar(f, bytes, memory_map, t)
            }
            TyKind::Adt(adt, _) if b.len() == 2 * size_of::<usize>() => match adt.0 {
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
                let Ok(layout) = f.db.layout_of_ty(t.clone(), trait_env) else {
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
        TyKind::Tuple(_, subst) => {
            let Ok(layout) = f.db.layout_of_ty(ty.clone(), trait_env.clone()) else {
                return f.write_str("<layout-error>");
            };
            f.write_str("(")?;
            let mut first = true;
            for (id, ty) in subst.iter(Interner).enumerate() {
                if first {
                    first = false;
                } else {
                    f.write_str(", ")?;
                }
                let ty = ty.assert_ty_ref(Interner); // Tuple only has type argument
                let offset = layout.fields.offset(id).bytes_usize();
                let Ok(layout) = f.db.layout_of_ty(ty.clone(), trait_env.clone()) else {
                    f.write_str("<layout-error>")?;
                    continue;
                };
                let size = layout.size.bytes_usize();
                render_const_scalar(f, &b[offset..offset + size], memory_map, ty)?;
            }
            f.write_str(")")
        }
        TyKind::Adt(adt, subst) => {
            let Ok(layout) = f.db.layout_of_adt(adt.0, subst.clone(), trait_env.clone()) else {
                return f.write_str("<layout-error>");
            };
            match adt.0 {
                hir_def::AdtId::StructId(s) => {
                    let data = f.db.struct_signature(s);
                    write!(f, "{}", data.name.display(f.db, f.edition()))?;
                    let field_types = f.db.field_types(s.into());
                    render_variant_after_name(
                        &f.db.variant_fields(s.into()),
                        f,
                        &field_types,
                        f.db.trait_environment(adt.0.into()),
                        &layout,
                        subst,
                        b,
                        memory_map,
                    )
                }
                hir_def::AdtId::UnionId(u) => {
                    write!(f, "{}", f.db.union_signature(u).name.display(f.db, f.edition()))
                }
                hir_def::AdtId::EnumId(e) => {
                    let Ok(target_data_layout) = f.db.target_data_layout(trait_env.krate) else {
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
                        f.db.enum_variants(loc.parent).variants[loc.index as usize]
                            .1
                            .display(f.db, f.edition())
                    )?;
                    let field_types = f.db.field_types(var_id.into());
                    render_variant_after_name(
                        &f.db.variant_fields(var_id.into()),
                        f,
                        &field_types,
                        f.db.trait_environment(adt.0.into()),
                        var_layout,
                        subst,
                        b,
                        memory_map,
                    )
                }
            }
        }
        TyKind::FnDef(..) => ty.hir_fmt(f),
        TyKind::Function(_) | TyKind::Raw(_, _) => {
            let it = u128::from_le_bytes(pad16(b, false));
            write!(f, "{it:#X} as ")?;
            ty.hir_fmt(f)
        }
        TyKind::Array(ty, len) => {
            let Some(len) = try_const_usize(f.db, len) else {
                return f.write_str("<unknown-array-len>");
            };
            let Ok(layout) = f.db.layout_of_ty(ty.clone(), trait_env) else {
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
        // The below arms are unreachable, since const eval will bail out before here.
        TyKind::Foreign(_) => f.write_str("<extern-type>"),
        TyKind::Error
        | TyKind::Placeholder(_)
        | TyKind::Alias(_)
        | TyKind::AssociatedType(_, _)
        | TyKind::OpaqueType(_, _)
        | TyKind::BoundVar(_)
        | TyKind::InferenceVar(_, _) => f.write_str("<placeholder-or-unknown-type>"),
        // The below arms are unreachable, since we handled them in ref case.
        TyKind::Slice(_) | TyKind::Str | TyKind::Dyn(_) => f.write_str("<unsized-value>"),
    }
}

fn render_variant_after_name(
    data: &VariantFields,
    f: &mut HirFormatter<'_>,
    field_types: &ArenaMap<LocalFieldId, Binders<Ty>>,
    trait_env: Arc<TraitEnvironment>,
    layout: &Layout,
    subst: &Substitution,
    b: &[u8],
    memory_map: &MemoryMap,
) -> Result<(), HirDisplayError> {
    match data.shape {
        FieldsShape::Record | FieldsShape::Tuple => {
            let render_field = |f: &mut HirFormatter<'_>, id: LocalFieldId| {
                let offset = layout.fields.offset(u32::from(id.into_raw()) as usize).bytes_usize();
                let ty = field_types[id].clone().substitute(Interner, subst);
                let Ok(layout) = f.db.layout_of_ty(ty.clone(), trait_env.clone()) else {
                    return f.write_str("<layout-error>");
                };
                let size = layout.size.bytes_usize();
                render_const_scalar(f, &b[offset..offset + size], memory_map, &ty)
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

impl HirDisplay for BoundVar {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "?{}.{}", self.debruijn.depth(), self.index)
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(
        &self,
        f @ &mut HirFormatter { db, .. }: &mut HirFormatter<'_>,
    ) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }

        match self.kind(Interner) {
            TyKind::Never => write!(f, "!")?,
            TyKind::Str => write!(f, "str")?,
            TyKind::Scalar(Scalar::Bool) => write!(f, "bool")?,
            TyKind::Scalar(Scalar::Char) => write!(f, "char")?,
            &TyKind::Scalar(Scalar::Float(t)) => write!(f, "{}", primitive::float_ty_to_string(t))?,
            &TyKind::Scalar(Scalar::Int(t)) => write!(f, "{}", primitive::int_ty_to_string(t))?,
            &TyKind::Scalar(Scalar::Uint(t)) => write!(f, "{}", primitive::uint_ty_to_string(t))?,
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
            kind @ (TyKind::Raw(m, t) | TyKind::Ref(m, _, t)) => {
                if let TyKind::Ref(_, l, _) = kind {
                    f.write_char('&')?;
                    if f.render_lifetime(l) {
                        l.hir_fmt(f)?;
                        f.write_char(' ')?;
                    }
                    match m {
                        Mutability::Not => (),
                        Mutability::Mut => f.write_str("mut ")?,
                    }
                } else {
                    write!(
                        f,
                        "*{}",
                        match m {
                            Mutability::Not => "const ",
                            Mutability::Mut => "mut ",
                        }
                    )?;
                }

                // FIXME: all this just to decide whether to use parentheses...
                let contains_impl_fn = |bounds: &[QuantifiedWhereClause]| {
                    bounds.iter().any(|bound| {
                        if let WhereClause::Implemented(trait_ref) = bound.skip_binders() {
                            let trait_ = trait_ref.hir_trait_id();
                            fn_traits(db, trait_).any(|it| it == trait_)
                        } else {
                            false
                        }
                    })
                };
                let (preds_to_print, has_impl_fn_pred) = match t.kind(Interner) {
                    TyKind::Dyn(dyn_ty) => {
                        let bounds = dyn_ty.bounds.skip_binders().interned();
                        let render_lifetime = f.render_lifetime(&dyn_ty.lifetime);
                        (bounds.len() + render_lifetime as usize, contains_impl_fn(bounds))
                    }
                    TyKind::Alias(AliasTy::Opaque(OpaqueTy {
                        opaque_ty_id,
                        substitution: parameters,
                    }))
                    | TyKind::OpaqueType(opaque_ty_id, parameters) => {
                        let impl_trait_id = db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
                        if let ImplTraitId::ReturnTypeImplTrait(func, idx) = impl_trait_id {
                            let datas = db
                                .return_type_impl_traits(func)
                                .expect("impl trait id without data");
                            let data =
                                (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                            let bounds = data.substitute(Interner, parameters);
                            let mut len = bounds.skip_binders().len();

                            // Don't count Sized but count when it absent
                            // (i.e. when explicit ?Sized bound is set).
                            let default_sized = SizedByDefault::Sized { anchor: func.krate(db) };
                            let sized_bounds = bounds
                                .skip_binders()
                                .iter()
                                .filter(|b| {
                                    matches!(
                                        b.skip_binders(),
                                        WhereClause::Implemented(trait_ref)
                                            if default_sized.is_sized_trait(
                                                trait_ref.hir_trait_id(),
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

                            (len, contains_impl_fn(bounds.skip_binders()))
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
            TyKind::Tuple(_, substs) => {
                if substs.len(Interner) == 1 {
                    write!(f, "(")?;
                    substs.at(Interner, 0).hir_fmt(f)?;
                    write!(f, ",)")?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(substs.as_slice(Interner), ", ")?;
                    write!(f, ")")?;
                }
            }
            TyKind::Function(fn_ptr) => {
                let sig = CallableSig::from_fn_ptr(fn_ptr);
                sig.hir_fmt(f)?;
            }
            TyKind::FnDef(def, parameters) => {
                let def = from_chalk(db, *def);
                let sig = db.callable_item_signature(def).substitute(Interner, parameters);

                if f.display_kind.is_source_code() {
                    // `FnDef` is anonymous and there's no surface syntax for it. Show it as a
                    // function pointer type.
                    return sig.hir_fmt(f);
                }
                if let Safety::Unsafe = sig.safety {
                    write!(f, "unsafe ")?;
                }
                if !matches!(sig.abi, FnAbi::Rust | FnAbi::RustCall) {
                    f.write_str("extern \"")?;
                    f.write_str(sig.abi.as_str())?;
                    f.write_str("\" ")?;
                }

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
                            db.enum_variants(loc.parent).variants[loc.index as usize]
                                .1
                                .display(db, f.edition())
                        )?
                    }
                };
                f.end_location_link();

                if parameters.len(Interner) > 0 {
                    let generic_def_id = GenericDefId::from_callable(db, def);
                    let generics = generics(db, generic_def_id);
                    let (parent_len, self_param, type_, const_, impl_, lifetime) =
                        generics.provenance_split();
                    let parameters = parameters.as_slice(Interner);
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
                f.write_joined(sig.params(), ", ")?;
                write!(f, ")")?;
                let ret = sig.ret();
                if !ret.is_unit() {
                    write!(f, " -> ")?;
                    ret.hir_fmt(f)?;
                }
            }
            TyKind::Adt(AdtId(def_id), parameters) => {
                f.start_location_link((*def_id).into());
                match f.display_kind {
                    DisplayKind::Diagnostics | DisplayKind::Test => {
                        let name = match *def_id {
                            hir_def::AdtId::StructId(it) => db.struct_signature(it).name.clone(),
                            hir_def::AdtId::UnionId(it) => db.union_signature(it).name.clone(),
                            hir_def::AdtId::EnumId(it) => db.enum_signature(it).name.clone(),
                        };
                        write!(f, "{}", name.display(f.db, f.edition()))?;
                    }
                    DisplayKind::SourceCode { target_module_id: module_id, allow_opaque: _ } => {
                        if let Some(path) = find_path::find_path(
                            db,
                            ItemInNs::Types((*def_id).into()),
                            module_id,
                            PrefixKind::Plain,
                            false,
                            // FIXME: no_std Cfg?
                            ImportPathConfig {
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

                let generic_def = self.as_generic_def(db);

                hir_fmt_generics(f, parameters.as_slice(Interner), generic_def, None)?;
            }
            TyKind::AssociatedType(assoc_type_id, parameters) => {
                let type_alias = from_assoc_type_id(*assoc_type_id);
                let trait_ = match type_alias.lookup(db).container {
                    ItemContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_data = db.trait_signature(trait_);
                let type_alias_data = db.type_alias_signature(type_alias);

                // Use placeholder associated types when the target is test (https://rust-lang.github.io/chalk/book/clauses/type_equality.html#placeholder-associated-types)
                if f.display_kind.is_test() {
                    f.start_location_link(trait_.into());
                    write!(f, "{}", trait_data.name.display(f.db, f.edition()))?;
                    f.end_location_link();
                    write!(f, "::")?;

                    f.start_location_link(type_alias.into());
                    write!(f, "{}", type_alias_data.name.display(f.db, f.edition()))?;
                    f.end_location_link();
                    // Note that the generic args for the associated type come before those for the
                    // trait (including the self type).
                    hir_fmt_generics(f, parameters.as_slice(Interner), None, None)
                } else {
                    let projection_ty = ProjectionTy {
                        associated_ty_id: to_assoc_type_id(type_alias),
                        substitution: parameters.clone(),
                    };

                    projection_ty.hir_fmt(f)
                }?;
            }
            TyKind::Foreign(type_alias) => {
                let alias = from_foreign_def_id(*type_alias);
                let type_alias = db.type_alias_signature(alias);
                f.start_location_link(alias.into());
                write!(f, "{}", type_alias.name.display(f.db, f.edition()))?;
                f.end_location_link();
            }
            TyKind::OpaqueType(opaque_ty_id, parameters) => {
                if !f.display_kind.allows_opaque() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::OpaqueType,
                    ));
                }
                let impl_trait_id = db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data =
                            (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &parameters);
                        let krate = func.krate(db);
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
                            Either::Left(self),
                            bounds.skip_binders(),
                            SizedByDefault::Sized { anchor: krate },
                        )?;
                        // FIXME: it would maybe be good to distinguish this from the alias type (when debug printing), and to show the substitution
                    }
                    ImplTraitId::TypeAliasImplTrait(alias, idx) => {
                        let datas =
                            db.type_alias_impl_traits(alias).expect("impl trait id without data");
                        let data = (*datas).as_ref().map(|it| it.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &parameters);
                        let krate = alias.krate(db);
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
                            Either::Left(self),
                            bounds.skip_binders(),
                            SizedByDefault::Sized { anchor: krate },
                        )?;
                    }
                    ImplTraitId::AsyncBlockTypeImplTrait(body, ..) => {
                        let future_trait =
                            LangItem::Future.resolve_trait(db, body.module(db).krate());
                        let output = future_trait.and_then(|t| {
                            db.trait_items(t)
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
                        parameters.at(Interner, 0).hir_fmt(f)?;
                        write!(f, ">")?;
                    }
                }
            }
            TyKind::Closure(id, substs) => {
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
                        return write!(f, "{{closure#{:?}}}", id.0.as_u32());
                    }
                    ClosureStyle::ClosureWithSubst => {
                        write!(f, "{{closure#{:?}}}", id.0.as_u32())?;
                        return hir_fmt_generics(f, substs.as_slice(Interner), None, None);
                    }
                    _ => (),
                }
                let sig = ClosureSubst(substs).sig_ty().callable_sig(db);
                if let Some(sig) = sig {
                    let InternedClosure(def, _) = db.lookup_intern_closure((*id).into());
                    let infer = db.infer(def);
                    let (_, kind) = infer.closure_info(id);
                    match f.closure_style {
                        ClosureStyle::ImplFn => write!(f, "impl {kind:?}(")?,
                        ClosureStyle::RANotation => write!(f, "|")?,
                        _ => unreachable!(),
                    }
                    if sig.params().is_empty() {
                    } else if f.should_truncate() {
                        write!(f, "{TYPE_HINT_TRUNCATION}")?;
                    } else {
                        f.write_joined(sig.params(), ", ")?;
                    };
                    match f.closure_style {
                        ClosureStyle::ImplFn => write!(f, ")")?,
                        ClosureStyle::RANotation => write!(f, "|")?,
                        _ => unreachable!(),
                    }
                    if f.closure_style == ClosureStyle::RANotation || !sig.ret().is_unit() {
                        write!(f, " -> ")?;
                        // FIXME: We display `AsyncFn` as `-> impl Future`, but this is hard to fix because
                        // we don't have a trait environment here, required to normalize `<Ret as Future>::Output`.
                        sig.ret().hir_fmt(f)?;
                    }
                } else {
                    write!(f, "{{closure}}")?;
                }
            }
            TyKind::Placeholder(idx) => {
                let id = from_placeholder_idx(db, *idx);
                let generics = generics(db, id.parent);
                let param_data = &generics[id.local_id];
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
                            let substs = generics.placeholder_subst(db);
                            let bounds = db
                                .generic_predicates(id.parent)
                                .iter()
                                .map(|pred| pred.clone().substitute(Interner, &substs))
                                .filter(|wc| match wc.skip_binders() {
                                    WhereClause::Implemented(tr) => {
                                        tr.self_type_parameter(Interner) == *self
                                    }
                                    WhereClause::AliasEq(AliasEq {
                                        alias: AliasTy::Projection(proj),
                                        ty: _,
                                    }) => proj.self_type_parameter(db) == *self,
                                    WhereClause::AliasEq(_) => false,
                                    WhereClause::TypeOutlives(to) => to.ty == *self,
                                    WhereClause::LifetimeOutlives(_) => false,
                                })
                                .collect::<Vec<_>>();
                            let krate = id.parent.module(db).krate();
                            write_bounds_like_dyn_trait_with_prefix(
                                f,
                                "impl",
                                Either::Left(self),
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
            TyKind::BoundVar(idx) => idx.hir_fmt(f)?,
            TyKind::Dyn(dyn_ty) => {
                // Reorder bounds to satisfy `write_bounds_like_dyn_trait()`'s expectation.
                // FIXME: `Iterator::partition_in_place()` or `Vec::extract_if()` may make it
                // more efficient when either of them hits stable.
                let mut bounds: SmallVec<[_; 4]> =
                    dyn_ty.bounds.skip_binders().iter(Interner).cloned().collect();
                let (auto_traits, others): (SmallVec<[_; 4]>, _) =
                    bounds.drain(1..).partition(|b| b.skip_binders().trait_id().is_some());
                bounds.extend(others);
                bounds.extend(auto_traits);

                if f.render_lifetime(&dyn_ty.lifetime) {
                    // we skip the binders in `write_bounds_like_dyn_trait_with_prefix`
                    bounds.push(Binders::empty(
                        Interner,
                        chalk_ir::WhereClause::TypeOutlives(chalk_ir::TypeOutlives {
                            ty: self.clone(),
                            lifetime: dyn_ty.lifetime.clone(),
                        }),
                    ));
                }

                write_bounds_like_dyn_trait_with_prefix(
                    f,
                    "dyn",
                    Either::Left(self),
                    &bounds,
                    SizedByDefault::NotSized,
                )?;
            }
            TyKind::Alias(AliasTy::Projection(p_ty)) => p_ty.hir_fmt(f)?,
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                if !f.display_kind.allows_opaque() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::OpaqueType,
                    ));
                }
                let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data =
                            (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &opaque_ty.substitution);
                        let krate = func.krate(db);
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
                            Either::Left(self),
                            bounds.skip_binders(),
                            SizedByDefault::Sized { anchor: krate },
                        )?;
                    }
                    ImplTraitId::TypeAliasImplTrait(alias, idx) => {
                        let datas =
                            db.type_alias_impl_traits(alias).expect("impl trait id without data");
                        let data =
                            (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &opaque_ty.substitution);
                        let krate = alias.krate(db);
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
                            Either::Left(self),
                            bounds.skip_binders(),
                            SizedByDefault::Sized { anchor: krate },
                        )?;
                    }
                    ImplTraitId::AsyncBlockTypeImplTrait(..) => {
                        write!(f, "{{async block}}")?;
                    }
                };
            }
            TyKind::Error => {
                if f.display_kind.is_source_code() {
                    f.write_char('_')?;
                } else {
                    write!(f, "{{unknown}}")?;
                }
            }
            TyKind::InferenceVar(..) => write!(f, "_")?,
            TyKind::Coroutine(_, subst) => {
                if f.display_kind.is_source_code() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::Coroutine,
                    ));
                }
                let subst = subst.as_slice(Interner);
                let a: Option<SmallVec<[&Ty; 3]>> = subst
                    .get(subst.len() - 3..)
                    .and_then(|args| args.iter().map(|arg| arg.ty(Interner)).collect());

                if let Some([resume_ty, yield_ty, ret_ty]) = a.as_deref() {
                    write!(f, "|")?;
                    resume_ty.hir_fmt(f)?;
                    write!(f, "|")?;

                    write!(f, " yields ")?;
                    yield_ty.hir_fmt(f)?;

                    write!(f, " -> ")?;
                    ret_ty.hir_fmt(f)?;
                } else {
                    // This *should* be unreachable, but fallback just in case.
                    write!(f, "{{coroutine}}")?;
                }
            }
            TyKind::CoroutineWitness(..) => write!(f, "{{coroutine witness}}")?,
        }
        Ok(())
    }
}

fn hir_fmt_generics(
    f: &mut HirFormatter<'_>,
    parameters: &[GenericArg],
    generic_def: Option<hir_def::GenericDefId>,
    self_: Option<&Ty>,
) -> Result<(), HirDisplayError> {
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

fn generic_args_sans_defaults<'ga>(
    f: &mut HirFormatter<'_>,
    generic_def: Option<hir_def::GenericDefId>,
    parameters: &'ga [GenericArg],
) -> &'ga [GenericArg] {
    if f.display_kind.is_source_code() || f.omit_verbose_types() {
        match generic_def
            .map(|generic_def_id| f.db.generic_defaults(generic_def_id))
            .filter(|it| !it.is_empty())
        {
            None => parameters,
            Some(default_parameters) => {
                let should_show = |arg: &GenericArg, i: usize| {
                    let is_err = |arg: &GenericArg| match arg.data(Interner) {
                        chalk_ir::GenericArgData::Lifetime(it) => {
                            *it.data(Interner) == LifetimeData::Error
                        }
                        chalk_ir::GenericArgData::Ty(it) => *it.kind(Interner) == TyKind::Error,
                        chalk_ir::GenericArgData::Const(it) => matches!(
                            it.data(Interner).value,
                            ConstValue::Concrete(ConcreteConst {
                                interned: ConstScalar::Unknown,
                                ..
                            })
                        ),
                    };
                    // if the arg is error like, render it to inform the user
                    if is_err(arg) {
                        return true;
                    }
                    // otherwise, if the arg is equal to the param default, hide it (unless the
                    // default is an error which can happen for the trait Self type)
                    match default_parameters.get(i) {
                        None => true,
                        Some(default_parameter) => {
                            // !is_err(default_parameter.skip_binders())
                            // &&
                            arg != &default_parameter.clone().substitute(Interner, &parameters[..i])
                        }
                    }
                };
                let mut default_from = 0;
                for (i, parameter) in parameters.iter().enumerate() {
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

fn hir_fmt_generic_arguments(
    f: &mut HirFormatter<'_>,
    parameters: &[GenericArg],
    self_: Option<&Ty>,
) -> Result<(), HirDisplayError> {
    let mut first = true;
    let lifetime_offset = parameters.iter().position(|arg| arg.lifetime(Interner).is_some());

    let (ty_or_const, lifetimes) = match lifetime_offset {
        Some(offset) => parameters.split_at(offset),
        None => (parameters, &[][..]),
    };
    for generic_arg in lifetimes.iter().chain(ty_or_const) {
        if !mem::take(&mut first) {
            write!(f, ", ")?;
        }
        match self_ {
            self_ @ Some(_) if generic_arg.ty(Interner) == self_ => write!(f, "Self")?,
            _ => generic_arg.hir_fmt(f)?,
        }
    }
    Ok(())
}

impl HirDisplay for CallableSig {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let CallableSig { params_and_return: _, is_varargs, safety, abi: _ } = *self;
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
        f.write_joined(self.params(), ", ")?;
        if is_varargs {
            if self.params().is_empty() {
                write!(f, "...")?;
            } else {
                write!(f, ", ...")?;
            }
        }
        write!(f, ")")?;
        let ret = self.ret();
        if !ret.is_unit() {
            write!(f, " -> ")?;
            ret.hir_fmt(f)?;
        }
        Ok(())
    }
}

fn fn_traits(db: &dyn DefDatabase, trait_: TraitId) -> impl Iterator<Item = TraitId> + '_ {
    let krate = trait_.lookup(db).container.krate();
    utils::fn_traits(db, krate)
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
                let sized_trait = LangItem::Sized.resolve_trait(db, anchor);
                Some(trait_) == sized_trait
            }
        }
    }
}

pub fn write_bounds_like_dyn_trait_with_prefix(
    f: &mut HirFormatter<'_>,
    prefix: &str,
    this: Either<&Ty, &Lifetime>,
    predicates: &[QuantifiedWhereClause],
    default_sized: SizedByDefault,
) -> Result<(), HirDisplayError> {
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

fn write_bounds_like_dyn_trait(
    f: &mut HirFormatter<'_>,
    this: Either<&Ty, &Lifetime>,
    predicates: &[QuantifiedWhereClause],
    default_sized: SizedByDefault,
) -> Result<(), HirDisplayError> {
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
    for p in predicates.iter() {
        match p.skip_binders() {
            WhereClause::Implemented(trait_ref) => {
                let trait_ = trait_ref.hir_trait_id();
                if default_sized.is_sized_trait(trait_, f.db) {
                    is_sized = true;
                    if matches!(default_sized, SizedByDefault::Sized { .. }) {
                        // Don't print +Sized, but rather +?Sized if absent.
                        continue;
                    }
                }
                if !is_fn_trait {
                    is_fn_trait = fn_traits(f.db, trait_).any(|it| it == trait_);
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
                    if let [self_, params @ ..] = trait_ref.substitution.as_slice(Interner) {
                        if let Some(args) =
                            params.first().and_then(|it| it.assert_ty_ref(Interner).as_tuple())
                        {
                            write!(f, "(")?;
                            hir_fmt_generic_arguments(
                                f,
                                args.as_slice(Interner),
                                self_.ty(Interner),
                            )?;
                            write!(f, ")")?;
                        }
                    }
                } else {
                    let params = generic_args_sans_defaults(
                        f,
                        Some(trait_.into()),
                        trait_ref.substitution.as_slice(Interner),
                    );
                    if let [self_, params @ ..] = params {
                        if !params.is_empty() {
                            write!(f, "<")?;
                            hir_fmt_generic_arguments(f, params, self_.ty(Interner))?;
                            // there might be assoc type bindings, so we leave the angle brackets open
                            angle_open = true;
                        }
                    }
                }
            }
            WhereClause::TypeOutlives(to) if Either::Left(&to.ty) == this => {
                if !is_fn_trait && angle_open {
                    write!(f, ">")?;
                    angle_open = false;
                }
                if !first {
                    write!(f, " + ")?;
                }
                to.lifetime.hir_fmt(f)?;
            }
            WhereClause::TypeOutlives(_) => {}
            WhereClause::LifetimeOutlives(lo) if Either::Right(&lo.a) == this => {
                if !is_fn_trait && angle_open {
                    write!(f, ">")?;
                    angle_open = false;
                }
                if !first {
                    write!(f, " + ")?;
                }
                lo.b.hir_fmt(f)?;
            }
            WhereClause::LifetimeOutlives(_) => {}
            WhereClause::AliasEq(alias_eq) if is_fn_trait => {
                is_fn_trait = false;
                if !alias_eq.ty.is_unit() {
                    write!(f, " -> ")?;
                    alias_eq.ty.hir_fmt(f)?;
                }
            }
            WhereClause::AliasEq(AliasEq { ty, alias }) => {
                // in types in actual Rust, these will always come
                // after the corresponding Implemented predicate
                if angle_open {
                    write!(f, ", ")?;
                } else {
                    write!(f, "<")?;
                    angle_open = true;
                }
                if let AliasTy::Projection(proj) = alias {
                    let assoc_ty_id = from_assoc_type_id(proj.associated_ty_id);
                    let type_alias = f.db.type_alias_signature(assoc_ty_id);
                    f.start_location_link(assoc_ty_id.into());
                    write!(f, "{}", type_alias.name.display(f.db, f.edition()))?;
                    f.end_location_link();

                    let proj_arg_count = generics(f.db, assoc_ty_id.into()).len_self();
                    let parent_len = proj.substitution.len(Interner) - proj_arg_count;
                    if proj_arg_count > 0 {
                        write!(f, "<")?;
                        hir_fmt_generic_arguments(
                            f,
                            &proj.substitution.as_slice(Interner)[parent_len..],
                            None,
                        )?;
                        write!(f, ">")?;
                    }
                    write!(f, " = ")?;
                }
                ty.hir_fmt(f)?;
            }
        }
        first = false;
    }
    if angle_open {
        write!(f, ">")?;
    }
    if let SizedByDefault::Sized { anchor } = default_sized {
        let sized_trait = LangItem::Sized.resolve_trait(f.db, anchor);
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

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let trait_ = self.hir_trait_id();
        f.start_location_link(trait_.into());
        write!(f, "{}", f.db.trait_signature(trait_).name.display(f.db, f.edition()))?;
        f.end_location_link();
        let substs = self.substitution.as_slice(Interner);
        hir_fmt_generics(f, &substs[1..], None, substs[0].ty(Interner))
    }
}

impl HirDisplay for WhereClause {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }

        match self {
            WhereClause::Implemented(trait_ref) => {
                trait_ref.self_type_parameter(Interner).hir_fmt(f)?;
                write!(f, ": ")?;
                trait_ref.hir_fmt(f)?;
            }
            WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
                write!(f, "<")?;
                let trait_ref = &projection_ty.trait_ref(f.db);
                trait_ref.self_type_parameter(Interner).hir_fmt(f)?;
                write!(f, " as ")?;
                trait_ref.hir_fmt(f)?;
                write!(f, ">::",)?;
                let type_alias = from_assoc_type_id(projection_ty.associated_ty_id);
                f.start_location_link(type_alias.into());
                write!(
                    f,
                    "{}",
                    f.db.type_alias_signature(type_alias).name.display(f.db, f.edition()),
                )?;
                f.end_location_link();
                write!(f, " = ")?;
                ty.hir_fmt(f)?;
            }
            WhereClause::AliasEq(_) => write!(f, "{{error}}")?,

            // FIXME implement these
            WhereClause::TypeOutlives(..) => {}
            WhereClause::LifetimeOutlives(..) => {}
        }
        Ok(())
    }
}

impl HirDisplay for LifetimeOutlives {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        self.a.hir_fmt(f)?;
        write!(f, ": ")?;
        self.b.hir_fmt(f)
    }
}

impl HirDisplay for Lifetime {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        self.interned().hir_fmt(f)
    }
}

impl HirDisplay for LifetimeData {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            LifetimeData::Placeholder(idx) => {
                let id = lt_from_placeholder_idx(f.db, *idx);
                let generics = generics(f.db, id.parent);
                let param_data = &generics[id.local_id];
                write!(f, "{}", param_data.name.display(f.db, f.edition()))?;
                Ok(())
            }
            LifetimeData::BoundVar(idx) => idx.hir_fmt(f),
            LifetimeData::InferenceVar(_) => write!(f, "_"),
            LifetimeData::Static => write!(f, "'static"),
            LifetimeData::Error => {
                if cfg!(test) {
                    write!(f, "'?")
                } else {
                    write!(f, "'_")
                }
            }
            LifetimeData::Erased => write!(f, "'<erased>"),
            LifetimeData::Phantom(void, _) => match *void {},
        }
    }
}

impl HirDisplay for DomainGoal {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            DomainGoal::Holds(wc) => {
                write!(f, "Holds(")?;
                wc.hir_fmt(f)?;
                write!(f, ")")?;
            }
            _ => write!(f, "_")?,
        }
        Ok(())
    }
}

pub fn write_visibility(
    module_id: ModuleId,
    vis: Visibility,
    f: &mut HirFormatter<'_>,
) -> Result<(), HirDisplayError> {
    match vis {
        Visibility::Public => write!(f, "pub "),
        Visibility::Module(vis_id, _) => {
            let def_map = module_id.def_map(f.db);
            let root_module_id = def_map.module_id(DefMap::ROOT);
            if vis_id == module_id {
                // pub(self) or omitted
                Ok(())
            } else if root_module_id == vis_id {
                write!(f, "pub(crate) ")
            } else if module_id.containing_module(f.db) == Some(vis_id) {
                write!(f, "pub(super) ")
            } else {
                write!(f, "pub(in ...) ")
            }
        }
    }
}

pub trait HirDisplayWithExpressionStore {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError>;
}

impl<T: ?Sized + HirDisplayWithExpressionStore> HirDisplayWithExpressionStore for &'_ T {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
        T::hir_fmt(&**self, f, store)
    }
}

pub fn hir_display_with_store<'a, T: HirDisplayWithExpressionStore + 'a>(
    value: T,
    store: &'a ExpressionStore,
) -> impl HirDisplay + 'a {
    ExpressionStoreAdapter(value, store)
}

struct ExpressionStoreAdapter<'a, T>(T, &'a ExpressionStore);

impl<'a, T> ExpressionStoreAdapter<'a, T> {
    fn wrap(store: &'a ExpressionStore) -> impl Fn(T) -> ExpressionStoreAdapter<'a, T> {
        move |value| ExpressionStoreAdapter(value, store)
    }
}

impl<T: HirDisplayWithExpressionStore> HirDisplay for ExpressionStoreAdapter<'_, T> {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        T::hir_fmt(&self.0, f, self.1)
    }
}
impl HirDisplayWithExpressionStore for LifetimeRefId {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
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

impl HirDisplayWithExpressionStore for TypeRefId {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
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

impl HirDisplayWithExpressionStore for ConstRef {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        _store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
        // FIXME
        write!(f, "{{const}}")?;

        Ok(())
    }
}

impl HirDisplayWithExpressionStore for TypeBound {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
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

impl HirDisplayWithExpressionStore for Path {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
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
                        if let Some(ret) = generic_args.bindings[0].type_ref {
                            if !matches!(&store[ret], TypeRef::Tuple(v) if v.is_empty()) {
                                write!(f, " -> ")?;
                                ret.hir_fmt(f, store)?;
                            }
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

impl HirDisplayWithExpressionStore for hir_def::expr_store::path::GenericArg {
    fn hir_fmt(
        &self,
        f: &mut HirFormatter<'_>,
        store: &ExpressionStore,
    ) -> Result<(), HirDisplayError> {
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

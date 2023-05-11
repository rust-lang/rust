//! The `HirDisplay` trait, which serves two purposes: Turning various bits from
//! HIR back into source code, and just displaying them for debugging/testing
//! purposes.

use std::fmt::{self, Debug};

use base_db::CrateId;
use chalk_ir::{BoundVar, TyKind};
use hir_def::{
    adt::VariantData,
    body,
    db::DefDatabase,
    find_path,
    generics::{TypeOrConstParamData, TypeParamProvenance},
    item_scope::ItemInNs,
    lang_item::{LangItem, LangItemTarget},
    path::{Path, PathKind},
    type_ref::{TraitBoundModifier, TypeBound, TypeRef},
    visibility::Visibility,
    HasModule, ItemContainerId, LocalFieldId, Lookup, ModuleDefId, ModuleId, TraitId,
};
use hir_expand::{hygiene::Hygiene, name::Name};
use intern::{Internable, Interned};
use itertools::Itertools;
use smallvec::SmallVec;

use crate::{
    db::HirDatabase,
    from_assoc_type_id, from_foreign_def_id, from_placeholder_idx,
    layout::layout_of_ty,
    lt_from_placeholder_idx,
    mapping::from_chalk,
    mir::pad16,
    primitive, to_assoc_type_id,
    utils::{self, generics},
    AdtId, AliasEq, AliasTy, Binders, CallableDefId, CallableSig, Const, ConstScalar, ConstValue,
    DomainGoal, GenericArg, ImplTraitId, Interner, Lifetime, LifetimeData, LifetimeOutlives,
    MemoryMap, Mutability, OpaqueTy, ProjectionTy, ProjectionTyExt, QuantifiedWhereClause, Scalar,
    Substitution, TraitRef, TraitRefExt, Ty, TyExt, WhereClause,
};

pub trait HirWrite: fmt::Write {
    fn start_location_link(&mut self, location: ModuleDefId);
    fn end_location_link(&mut self);
}

// String will ignore link metadata
impl HirWrite for String {
    fn start_location_link(&mut self, _: ModuleDefId) {}

    fn end_location_link(&mut self) {}
}

// `core::Formatter` will ignore metadata
impl HirWrite for fmt::Formatter<'_> {
    fn start_location_link(&mut self, _: ModuleDefId) {}
    fn end_location_link(&mut self) {}
}

pub struct HirFormatter<'a> {
    pub db: &'a dyn HirDatabase,
    fmt: &'a mut dyn HirWrite,
    buf: String,
    curr_size: usize,
    pub(crate) max_size: Option<usize>,
    omit_verbose_types: bool,
    display_target: DisplayTarget,
}

impl HirFormatter<'_> {
    fn start_location_link(&mut self, location: ModuleDefId) {
        self.fmt.start_location_link(location);
    }

    fn end_location_link(&mut self) {
        self.fmt.end_location_link();
    }
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError>;

    /// Returns a `Display`able type that is human-readable.
    fn into_displayable<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        max_size: Option<usize>,
        omit_verbose_types: bool,
        display_target: DisplayTarget,
    ) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        assert!(
            !matches!(display_target, DisplayTarget::SourceCode { .. }),
            "HirDisplayWrapper cannot fail with DisplaySourceCodeError, use HirDisplay::hir_fmt directly instead"
        );
        HirDisplayWrapper { db, t: self, max_size, omit_verbose_types, display_target }
    }

    /// Returns a `Display`able type that is human-readable.
    /// Use this for showing types to the user (e.g. diagnostics)
    fn display<'a>(&'a self, db: &'a dyn HirDatabase) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            omit_verbose_types: false,
            display_target: DisplayTarget::Diagnostics,
        }
    }

    /// Returns a `Display`able type that is human-readable and tries to be succinct.
    /// Use this for showing types to the user where space is constrained (e.g. doc popups)
    fn display_truncated<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        max_size: Option<usize>,
    ) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size,
            omit_verbose_types: true,
            display_target: DisplayTarget::Diagnostics,
        }
    }

    /// Returns a String representation of `self` that can be inserted into the given module.
    /// Use this when generating code (e.g. assists)
    fn display_source_code<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        module_id: ModuleId,
    ) -> Result<String, DisplaySourceCodeError> {
        let mut result = String::new();
        match self.hir_fmt(&mut HirFormatter {
            db,
            fmt: &mut result,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: None,
            omit_verbose_types: false,
            display_target: DisplayTarget::SourceCode { module_id },
        }) {
            Ok(()) => {}
            Err(HirDisplayError::FmtError) => panic!("Writing to String can't fail!"),
            Err(HirDisplayError::DisplaySourceCodeError(e)) => return Err(e),
        };
        Ok(result)
    }

    /// Returns a String representation of `self` for test purposes
    fn display_test<'a>(&'a self, db: &'a dyn HirDatabase) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper {
            db,
            t: self,
            max_size: None,
            omit_verbose_types: false,
            display_target: DisplayTarget::Test,
        }
    }
}

impl<'a> HirFormatter<'a> {
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
}

#[derive(Clone, Copy)]
pub enum DisplayTarget {
    /// Display types for inlays, doc popups, autocompletion, etc...
    /// Showing `{unknown}` or not qualifying paths is fine here.
    /// There's no reason for this to fail.
    Diagnostics,
    /// Display types for inserting them in source files.
    /// The generated code should compile, so paths need to be qualified.
    SourceCode { module_id: ModuleId },
    /// Only for test purpose to keep real types
    Test,
}

impl DisplayTarget {
    fn is_source_code(&self) -> bool {
        matches!(self, Self::SourceCode { .. })
    }
    fn is_test(&self) -> bool {
        matches!(self, Self::Test)
    }
}

#[derive(Debug)]
pub enum DisplaySourceCodeError {
    PathNotFound,
    UnknownType,
    Closure,
    Generator,
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
    omit_verbose_types: bool,
    display_target: DisplayTarget,
}

impl<T: HirDisplay> HirDisplayWrapper<'_, T> {
    pub fn write_to<F: HirWrite>(&self, f: &mut F) -> Result<(), HirDisplayError> {
        self.t.hir_fmt(&mut HirFormatter {
            db: self.db,
            fmt: f,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: self.max_size,
            omit_verbose_types: self.omit_verbose_types,
            display_target: self.display_target,
        })
    }
}

impl<'a, T> fmt::Display for HirDisplayWrapper<'a, T>
where
    T: HirDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.write_to(f) {
            Ok(()) => Ok(()),
            Err(HirDisplayError::FmtError) => Err(fmt::Error),
            Err(HirDisplayError::DisplaySourceCodeError(_)) => {
                // This should never happen
                panic!("HirDisplay::hir_fmt failed with DisplaySourceCodeError when calling Display::fmt!")
            }
        }
    }
}

const TYPE_HINT_TRUNCATION: &str = "…";

impl<T: HirDisplay> HirDisplay for &'_ T {
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
        write!(f, "<")?;
        fmt_trait_ref(f, &trait_ref, true)?;
        write!(f, ">::{}", f.db.type_alias_data(from_assoc_type_id(self.associated_ty_id)).name)?;
        let proj_params_count =
            self.substitution.len(Interner) - trait_ref.substitution.len(Interner);
        let proj_params = &self.substitution.as_slice(Interner)[..proj_params_count];
        if !proj_params.is_empty() {
            write!(f, "<")?;
            f.write_joined(proj_params, ", ")?;
            write!(f, ">")?;
        }
        Ok(())
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
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.type_or_consts[id.local_id];
                write!(f, "{}", param_data.name().unwrap())
            }
            ConstValue::Concrete(c) => match &c.interned {
                ConstScalar::Bytes(b, m) => render_const_scalar(f, &b, m, &data.ty),
                ConstScalar::Unknown => f.write_char('_'),
            },
        }
    }
}

pub struct HexifiedConst(pub Const);

impl HirDisplay for HexifiedConst {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        let data = &self.0.data(Interner);
        if let TyKind::Scalar(s) = data.ty.kind(Interner) {
            if matches!(s, Scalar::Int(_) | Scalar::Uint(_)) {
                if let ConstValue::Concrete(c) = &data.value {
                    if let ConstScalar::Bytes(b, m) = &c.interned {
                        let value = u128::from_le_bytes(pad16(b, false));
                        if value >= 10 {
                            render_const_scalar(f, &b, m, &data.ty)?;
                            return write!(f, " ({:#X})", value);
                        }
                    }
                }
            }
        }
        self.0.hir_fmt(f)
    }
}

fn render_const_scalar(
    f: &mut HirFormatter<'_>,
    b: &[u8],
    memory_map: &MemoryMap,
    ty: &Ty,
) -> Result<(), HirDisplayError> {
    match ty.kind(Interner) {
        chalk_ir::TyKind::Scalar(s) => match s {
            Scalar::Bool => write!(f, "{}", if b[0] == 0 { false } else { true }),
            Scalar::Char => {
                let x = u128::from_le_bytes(pad16(b, false)) as u32;
                let Ok(c) = char::try_from(x) else {
                    return f.write_str("<unicode-error>");
                };
                write!(f, "{c:?}")
            }
            Scalar::Int(_) => {
                let x = i128::from_le_bytes(pad16(b, true));
                write!(f, "{x}")
            }
            Scalar::Uint(_) => {
                let x = u128::from_le_bytes(pad16(b, false));
                write!(f, "{x}")
            }
            Scalar::Float(fl) => match fl {
                chalk_ir::FloatTy::F32 => {
                    let x = f32::from_le_bytes(b.try_into().unwrap());
                    write!(f, "{x:?}")
                }
                chalk_ir::FloatTy::F64 => {
                    let x = f64::from_le_bytes(b.try_into().unwrap());
                    write!(f, "{x:?}")
                }
            },
        },
        chalk_ir::TyKind::Ref(_, _, t) => match t.kind(Interner) {
            chalk_ir::TyKind::Str => {
                let addr = usize::from_le_bytes(b[0..b.len() / 2].try_into().unwrap());
                let bytes = memory_map.0.get(&addr).map(|x| &**x).unwrap_or(&[]);
                let s = std::str::from_utf8(bytes).unwrap_or("<utf8-error>");
                write!(f, "{s:?}")
            }
            _ => f.write_str("<ref-not-supported>"),
        },
        chalk_ir::TyKind::Tuple(_, subst) => {
            // FIXME: Remove this line. If the target data layout is independent
            // of the krate, the `db.target_data_layout` and its callers like `layout_of_ty` don't need
            // to get krate. Otherwise, we need to get krate from the final callers of the hir display
            // infrastructure and have it here as a field on `f`.
            let krate = *f.db.crate_graph().crates_in_topological_order().last().unwrap();
            let Ok(layout) = layout_of_ty(f.db, ty, krate) else {
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
                let Ok(layout) = layout_of_ty(f.db, &ty, krate) else {
                    f.write_str("<layout-error>")?;
                    continue;
                };
                let size = layout.size.bytes_usize();
                render_const_scalar(f, &b[offset..offset + size], memory_map, &ty)?;
            }
            f.write_str(")")
        }
        chalk_ir::TyKind::Adt(adt, subst) => match adt.0 {
            hir_def::AdtId::StructId(s) => {
                let data = f.db.struct_data(s);
                let Ok(layout) = f.db.layout_of_adt(adt.0, subst.clone()) else {
                    return f.write_str("<layout-error>");
                };
                match data.variant_data.as_ref() {
                    VariantData::Record(fields) | VariantData::Tuple(fields) => {
                        let field_types = f.db.field_types(s.into());
                        let krate = adt.0.module(f.db.upcast()).krate();
                        let render_field = |f: &mut HirFormatter<'_>, id: LocalFieldId| {
                            let offset = layout
                                .fields
                                .offset(u32::from(id.into_raw()) as usize)
                                .bytes_usize();
                            let ty = field_types[id].clone().substitute(Interner, subst);
                            let Ok(layout) = layout_of_ty(f.db, &ty, krate) else {
                                return f.write_str("<layout-error>");
                            };
                            let size = layout.size.bytes_usize();
                            render_const_scalar(f, &b[offset..offset + size], memory_map, &ty)
                        };
                        let mut it = fields.iter();
                        if matches!(data.variant_data.as_ref(), VariantData::Record(_)) {
                            write!(f, "{} {{", data.name)?;
                            if let Some((id, data)) = it.next() {
                                write!(f, " {}: ", data.name)?;
                                render_field(f, id)?;
                            }
                            for (id, data) in it {
                                write!(f, ", {}: ", data.name)?;
                                render_field(f, id)?;
                            }
                            write!(f, " }}")?;
                        } else {
                            let mut it = it.map(|x| x.0);
                            write!(f, "{}(", data.name)?;
                            if let Some(id) = it.next() {
                                render_field(f, id)?;
                            }
                            for id in it {
                                write!(f, ", ")?;
                                render_field(f, id)?;
                            }
                            write!(f, ")")?;
                        }
                        return Ok(());
                    }
                    VariantData::Unit => write!(f, "{}", data.name),
                }
            }
            hir_def::AdtId::UnionId(u) => write!(f, "{}", f.db.union_data(u).name),
            hir_def::AdtId::EnumId(_) => f.write_str("<enum-not-supported>"),
        },
        chalk_ir::TyKind::FnDef(..) => ty.hir_fmt(f),
        _ => f.write_str("<not-supported>"),
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
            TyKind::Raw(m, t) | TyKind::Ref(m, _, t) => {
                if matches!(self.kind(Interner), TyKind::Raw(..)) {
                    write!(
                        f,
                        "*{}",
                        match m {
                            Mutability::Not => "const ",
                            Mutability::Mut => "mut ",
                        }
                    )?;
                } else {
                    write!(
                        f,
                        "&{}",
                        match m {
                            Mutability::Not => "",
                            Mutability::Mut => "mut ",
                        }
                    )?;
                }

                // FIXME: all this just to decide whether to use parentheses...
                let contains_impl_fn = |bounds: &[QuantifiedWhereClause]| {
                    bounds.iter().any(|bound| {
                        if let WhereClause::Implemented(trait_ref) = bound.skip_binders() {
                            let trait_ = trait_ref.hir_trait_id();
                            fn_traits(db.upcast(), trait_).any(|it| it == trait_)
                        } else {
                            false
                        }
                    })
                };
                let (preds_to_print, has_impl_fn_pred) = match t.kind(Interner) {
                    TyKind::Dyn(dyn_ty) if dyn_ty.bounds.skip_binders().interned().len() > 1 => {
                        let bounds = dyn_ty.bounds.skip_binders().interned();
                        (bounds.len(), contains_impl_fn(bounds))
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
                            let default_sized = SizedByDefault::Sized {
                                anchor: func.lookup(db.upcast()).module(db.upcast()).krate(),
                            };
                            let sized_bounds = bounds
                                .skip_binders()
                                .iter()
                                .filter(|b| {
                                    matches!(
                                        b.skip_binders(),
                                        WhereClause::Implemented(trait_ref)
                                            if default_sized.is_sized_trait(
                                                trait_ref.hir_trait_id(),
                                                db.upcast(),
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
                    f.write_joined(&*substs.as_slice(Interner), ", ")?;
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
                f.start_location_link(def.into());
                match def {
                    CallableDefId::FunctionId(ff) => write!(f, "fn {}", db.function_data(ff).name)?,
                    CallableDefId::StructId(s) => write!(f, "{}", db.struct_data(s).name)?,
                    CallableDefId::EnumVariantId(e) => {
                        write!(f, "{}", db.enum_data(e.parent).variants[e.local_id].name)?
                    }
                };
                f.end_location_link();
                if parameters.len(Interner) > 0 {
                    let generics = generics(db.upcast(), def.into());
                    let (parent_params, self_param, type_params, const_params, _impl_trait_params) =
                        generics.provenance_split();
                    let total_len = parent_params + self_param + type_params + const_params;
                    // We print all params except implicit impl Trait params. Still a bit weird; should we leave out parent and self?
                    if total_len > 0 {
                        // `parameters` are in the order of fn's params (including impl traits),
                        // parent's params (those from enclosing impl or trait, if any).
                        let parameters = parameters.as_slice(Interner);
                        let fn_params_len = self_param + type_params + const_params;
                        let fn_params = parameters.get(..fn_params_len);
                        let parent_params = parameters.get(parameters.len() - parent_params..);
                        let params = parent_params.into_iter().chain(fn_params).flatten();
                        write!(f, "<")?;
                        f.write_joined(params, ", ")?;
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
                match f.display_target {
                    DisplayTarget::Diagnostics | DisplayTarget::Test => {
                        let name = match *def_id {
                            hir_def::AdtId::StructId(it) => db.struct_data(it).name.clone(),
                            hir_def::AdtId::UnionId(it) => db.union_data(it).name.clone(),
                            hir_def::AdtId::EnumId(it) => db.enum_data(it).name.clone(),
                        };
                        write!(f, "{name}")?;
                    }
                    DisplayTarget::SourceCode { module_id } => {
                        if let Some(path) = find_path::find_path(
                            db.upcast(),
                            ItemInNs::Types((*def_id).into()),
                            module_id,
                            false,
                        ) {
                            write!(f, "{path}")?;
                        } else {
                            return Err(HirDisplayError::DisplaySourceCodeError(
                                DisplaySourceCodeError::PathNotFound,
                            ));
                        }
                    }
                }
                f.end_location_link();

                if parameters.len(Interner) > 0 {
                    let parameters_to_write = if f.display_target.is_source_code()
                        || f.omit_verbose_types()
                    {
                        match self
                            .as_generic_def(db)
                            .map(|generic_def_id| db.generic_defaults(generic_def_id))
                            .filter(|defaults| !defaults.is_empty())
                        {
                            None => parameters.as_slice(Interner),
                            Some(default_parameters) => {
                                fn should_show(
                                    parameter: &GenericArg,
                                    default_parameters: &[Binders<GenericArg>],
                                    i: usize,
                                    parameters: &Substitution,
                                ) -> bool {
                                    if parameter.ty(Interner).map(|x| x.kind(Interner))
                                        == Some(&TyKind::Error)
                                    {
                                        return true;
                                    }
                                    if let Some(ConstValue::Concrete(c)) = parameter
                                        .constant(Interner)
                                        .map(|x| &x.data(Interner).value)
                                    {
                                        if c.interned == ConstScalar::Unknown {
                                            return true;
                                        }
                                    }
                                    let default_parameter = match default_parameters.get(i) {
                                        Some(x) => x,
                                        None => return true,
                                    };
                                    let actual_default =
                                        default_parameter.clone().substitute(Interner, &parameters);
                                    parameter != &actual_default
                                }
                                let mut default_from = 0;
                                for (i, parameter) in parameters.iter(Interner).enumerate() {
                                    if should_show(parameter, &default_parameters, i, parameters) {
                                        default_from = i + 1;
                                    }
                                }
                                &parameters.as_slice(Interner)[0..default_from]
                            }
                        }
                    } else {
                        parameters.as_slice(Interner)
                    };
                    if !parameters_to_write.is_empty() {
                        write!(f, "<")?;

                        if f.display_target.is_source_code() {
                            let mut first = true;
                            for generic_arg in parameters_to_write {
                                if !first {
                                    write!(f, ", ")?;
                                }
                                first = false;

                                if generic_arg.ty(Interner).map(|ty| ty.kind(Interner))
                                    == Some(&TyKind::Error)
                                {
                                    write!(f, "_")?;
                                } else {
                                    generic_arg.hir_fmt(f)?;
                                }
                            }
                        } else {
                            f.write_joined(parameters_to_write, ", ")?;
                        }

                        write!(f, ">")?;
                    }
                }
            }
            TyKind::AssociatedType(assoc_type_id, parameters) => {
                let type_alias = from_assoc_type_id(*assoc_type_id);
                let trait_ = match type_alias.lookup(db.upcast()).container {
                    ItemContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_data = db.trait_data(trait_);
                let type_alias_data = db.type_alias_data(type_alias);

                // Use placeholder associated types when the target is test (https://rust-lang.github.io/chalk/book/clauses/type_equality.html#placeholder-associated-types)
                if f.display_target.is_test() {
                    f.start_location_link(trait_.into());
                    write!(f, "{}", trait_data.name)?;
                    f.end_location_link();
                    write!(f, "::")?;

                    f.start_location_link(type_alias.into());
                    write!(f, "{}", type_alias_data.name)?;
                    f.end_location_link();
                    // Note that the generic args for the associated type come before those for the
                    // trait (including the self type).
                    // FIXME: reconsider the generic args order upon formatting?
                    if parameters.len(Interner) > 0 {
                        write!(f, "<")?;
                        f.write_joined(parameters.as_slice(Interner), ", ")?;
                        write!(f, ">")?;
                    }
                } else {
                    let projection_ty = ProjectionTy {
                        associated_ty_id: to_assoc_type_id(type_alias),
                        substitution: parameters.clone(),
                    };

                    projection_ty.hir_fmt(f)?;
                }
            }
            TyKind::Foreign(type_alias) => {
                let alias = from_foreign_def_id(*type_alias);
                let type_alias = db.type_alias_data(alias);
                f.start_location_link(alias.into());
                write!(f, "{}", type_alias.name)?;
                f.end_location_link();
            }
            TyKind::OpaqueType(opaque_ty_id, parameters) => {
                let impl_trait_id = db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data =
                            (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &parameters);
                        let krate = func.lookup(db.upcast()).module(db.upcast()).krate();
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
                            bounds.skip_binders(),
                            SizedByDefault::Sized { anchor: krate },
                        )?;
                        // FIXME: it would maybe be good to distinguish this from the alias type (when debug printing), and to show the substitution
                    }
                    ImplTraitId::AsyncBlockTypeImplTrait(body, ..) => {
                        let future_trait = db
                            .lang_item(body.module(db.upcast()).krate(), LangItem::Future)
                            .and_then(LangItemTarget::as_trait);
                        let output = future_trait.and_then(|t| {
                            db.trait_data(t).associated_type_by_name(&hir_expand::name!(Output))
                        });
                        write!(f, "impl ")?;
                        if let Some(t) = future_trait {
                            f.start_location_link(t.into());
                        }
                        write!(f, "Future")?;
                        if let Some(_) = future_trait {
                            f.end_location_link();
                        }
                        write!(f, "<")?;
                        if let Some(t) = output {
                            f.start_location_link(t.into());
                        }
                        write!(f, "Output")?;
                        if let Some(_) = output {
                            f.end_location_link();
                        }
                        write!(f, " = ")?;
                        parameters.at(Interner, 0).hir_fmt(f)?;
                        write!(f, ">")?;
                    }
                }
            }
            TyKind::Closure(.., substs) => {
                if f.display_target.is_source_code() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::Closure,
                    ));
                }
                let sig = substs.at(Interner, 0).assert_ty_ref(Interner).callable_sig(db);
                if let Some(sig) = sig {
                    if sig.params().is_empty() {
                        write!(f, "||")?;
                    } else if f.should_truncate() {
                        write!(f, "|{TYPE_HINT_TRUNCATION}|")?;
                    } else {
                        write!(f, "|")?;
                        f.write_joined(sig.params(), ", ")?;
                        write!(f, "|")?;
                    };

                    write!(f, " -> ")?;
                    sig.ret().hir_fmt(f)?;
                } else {
                    write!(f, "{{closure}}")?;
                }
            }
            TyKind::Placeholder(idx) => {
                let id = from_placeholder_idx(db, *idx);
                let generics = generics(db.upcast(), id.parent);
                let param_data = &generics.params.type_or_consts[id.local_id];
                match param_data {
                    TypeOrConstParamData::TypeParamData(p) => match p.provenance {
                        TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                            write!(f, "{}", p.name.clone().unwrap_or_else(Name::missing))?
                        }
                        TypeParamProvenance::ArgumentImplTrait => {
                            let substs = generics.placeholder_subst(db);
                            let bounds = db
                                .generic_predicates(id.parent)
                                .iter()
                                .map(|pred| pred.clone().substitute(Interner, &substs))
                                .filter(|wc| match &wc.skip_binders() {
                                    WhereClause::Implemented(tr) => {
                                        &tr.self_type_parameter(Interner) == self
                                    }
                                    WhereClause::AliasEq(AliasEq {
                                        alias: AliasTy::Projection(proj),
                                        ty: _,
                                    }) => &proj.self_type_parameter(db) == self,
                                    _ => false,
                                })
                                .collect::<Vec<_>>();
                            let krate = id.parent.module(db.upcast()).krate();
                            write_bounds_like_dyn_trait_with_prefix(
                                f,
                                "impl",
                                &bounds,
                                SizedByDefault::Sized { anchor: krate },
                            )?;
                        }
                    },
                    TypeOrConstParamData::ConstParamData(p) => {
                        write!(f, "{}", p.name)?;
                    }
                }
            }
            TyKind::BoundVar(idx) => idx.hir_fmt(f)?,
            TyKind::Dyn(dyn_ty) => {
                // Reorder bounds to satisfy `write_bounds_like_dyn_trait()`'s expectation.
                // FIXME: `Iterator::partition_in_place()` or `Vec::drain_filter()` may make it
                // more efficient when either of them hits stable.
                let mut bounds: SmallVec<[_; 4]> =
                    dyn_ty.bounds.skip_binders().iter(Interner).cloned().collect();
                let (auto_traits, others): (SmallVec<[_; 4]>, _) =
                    bounds.drain(1..).partition(|b| b.skip_binders().trait_id().is_some());
                bounds.extend(others);
                bounds.extend(auto_traits);

                write_bounds_like_dyn_trait_with_prefix(
                    f,
                    "dyn",
                    &bounds,
                    SizedByDefault::NotSized,
                )?;
            }
            TyKind::Alias(AliasTy::Projection(p_ty)) => p_ty.hir_fmt(f)?,
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let impl_trait_id = db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data =
                            (*datas).as_ref().map(|rpit| rpit.impl_traits[idx].bounds.clone());
                        let bounds = data.substitute(Interner, &opaque_ty.substitution);
                        let krate = func.lookup(db.upcast()).module(db.upcast()).krate();
                        write_bounds_like_dyn_trait_with_prefix(
                            f,
                            "impl",
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
                if f.display_target.is_source_code() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::UnknownType,
                    ));
                }
                write!(f, "{{unknown}}")?;
            }
            TyKind::InferenceVar(..) => write!(f, "_")?,
            TyKind::Generator(_, subst) => {
                if f.display_target.is_source_code() {
                    return Err(HirDisplayError::DisplaySourceCodeError(
                        DisplaySourceCodeError::Generator,
                    ));
                }
                let subst = subst.as_slice(Interner);
                let a: Option<SmallVec<[&Ty; 3]>> = subst
                    .get(subst.len() - 3..)
                    .map(|args| args.iter().map(|arg| arg.ty(Interner)).collect())
                    .flatten();

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
                    write!(f, "{{generator}}")?;
                }
            }
            TyKind::GeneratorWitness(..) => write!(f, "{{generator witness}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for CallableSig {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        write!(f, "fn(")?;
        f.write_joined(self.params(), ", ")?;
        if self.is_varargs {
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
    Sized { anchor: CrateId },
}

impl SizedByDefault {
    fn is_sized_trait(self, trait_: TraitId, db: &dyn DefDatabase) -> bool {
        match self {
            Self::NotSized => false,
            Self::Sized { anchor } => {
                let sized_trait = db
                    .lang_item(anchor, LangItem::Sized)
                    .and_then(|lang_item| lang_item.as_trait());
                Some(trait_) == sized_trait
            }
        }
    }
}

pub fn write_bounds_like_dyn_trait_with_prefix(
    f: &mut HirFormatter<'_>,
    prefix: &str,
    predicates: &[QuantifiedWhereClause],
    default_sized: SizedByDefault,
) -> Result<(), HirDisplayError> {
    write!(f, "{prefix}")?;
    if !predicates.is_empty()
        || predicates.is_empty() && matches!(default_sized, SizedByDefault::Sized { .. })
    {
        write!(f, " ")?;
        write_bounds_like_dyn_trait(f, predicates, default_sized)
    } else {
        Ok(())
    }
}

fn write_bounds_like_dyn_trait(
    f: &mut HirFormatter<'_>,
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
                if default_sized.is_sized_trait(trait_, f.db.upcast()) {
                    is_sized = true;
                    if matches!(default_sized, SizedByDefault::Sized { .. }) {
                        // Don't print +Sized, but rather +?Sized if absent.
                        continue;
                    }
                }
                if !is_fn_trait {
                    is_fn_trait = fn_traits(f.db.upcast(), trait_).any(|it| it == trait_);
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
                write!(f, "{}", f.db.trait_data(trait_).name)?;
                f.end_location_link();
                if let [_, params @ ..] = &*trait_ref.substitution.as_slice(Interner) {
                    if is_fn_trait {
                        if let Some(args) =
                            params.first().and_then(|it| it.assert_ty_ref(Interner).as_tuple())
                        {
                            write!(f, "(")?;
                            f.write_joined(args.as_slice(Interner), ", ")?;
                            write!(f, ")")?;
                        }
                    } else if !params.is_empty() {
                        write!(f, "<")?;
                        f.write_joined(params, ", ")?;
                        // there might be assoc type bindings, so we leave the angle brackets open
                        angle_open = true;
                    }
                }
            }
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
                    let type_alias = f.db.type_alias_data(assoc_ty_id);
                    f.start_location_link(assoc_ty_id.into());
                    write!(f, "{}", type_alias.name)?;
                    f.end_location_link();

                    let proj_arg_count = generics(f.db.upcast(), assoc_ty_id.into()).len_self();
                    if proj_arg_count > 0 {
                        write!(f, "<")?;
                        f.write_joined(
                            &proj.substitution.as_slice(Interner)[..proj_arg_count],
                            ", ",
                        )?;
                        write!(f, ">")?;
                    }
                    write!(f, " = ")?;
                }
                ty.hir_fmt(f)?;
            }

            // FIXME implement these
            WhereClause::LifetimeOutlives(_) => {}
            WhereClause::TypeOutlives(_) => {}
        }
        first = false;
    }
    if angle_open {
        write!(f, ">")?;
    }
    if let SizedByDefault::Sized { anchor } = default_sized {
        let sized_trait =
            f.db.lang_item(anchor, LangItem::Sized).and_then(|lang_item| lang_item.as_trait());
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
        if let Some(_) = sized_trait {
            f.end_location_link();
        }
    }
    Ok(())
}

fn fmt_trait_ref(
    f: &mut HirFormatter<'_>,
    tr: &TraitRef,
    use_as: bool,
) -> Result<(), HirDisplayError> {
    if f.should_truncate() {
        return write!(f, "{TYPE_HINT_TRUNCATION}");
    }

    tr.self_type_parameter(Interner).hir_fmt(f)?;
    if use_as {
        write!(f, " as ")?;
    } else {
        write!(f, ": ")?;
    }
    let trait_ = tr.hir_trait_id();
    f.start_location_link(trait_.into());
    write!(f, "{}", f.db.trait_data(trait_).name)?;
    f.end_location_link();
    if tr.substitution.len(Interner) > 1 {
        write!(f, "<")?;
        f.write_joined(&tr.substitution.as_slice(Interner)[1..], ", ")?;
        write!(f, ">")?;
    }
    Ok(())
}

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        fmt_trait_ref(f, self, false)
    }
}

impl HirDisplay for WhereClause {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{TYPE_HINT_TRUNCATION}");
        }

        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.hir_fmt(f)?,
            WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
                write!(f, "<")?;
                fmt_trait_ref(f, &projection_ty.trait_ref(f.db), true)?;
                write!(f, ">::",)?;
                let type_alias = from_assoc_type_id(projection_ty.associated_ty_id);
                f.start_location_link(type_alias.into());
                write!(f, "{}", f.db.type_alias_data(type_alias).name,)?;
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
            LifetimeData::BoundVar(idx) => idx.hir_fmt(f),
            LifetimeData::InferenceVar(_) => write!(f, "_"),
            LifetimeData::Placeholder(idx) => {
                let id = lt_from_placeholder_idx(f.db, *idx);
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.lifetimes[id.local_id];
                write!(f, "{}", param_data.name)
            }
            LifetimeData::Static => write!(f, "'static"),
            LifetimeData::Erased => Ok(()),
            LifetimeData::Phantom(_, _) => Ok(()),
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
            _ => write!(f, "?")?,
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
        Visibility::Module(vis_id) => {
            let def_map = module_id.def_map(f.db.upcast());
            let root_module_id = def_map.module_id(def_map.root());
            if vis_id == module_id {
                // pub(self) or omitted
                Ok(())
            } else if root_module_id == vis_id {
                write!(f, "pub(crate) ")
            } else if module_id.containing_module(f.db.upcast()) == Some(vis_id) {
                write!(f, "pub(super) ")
            } else {
                write!(f, "pub(in ...) ")
            }
        }
    }
}

impl HirDisplay for TypeRef {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            TypeRef::Never => write!(f, "!")?,
            TypeRef::Placeholder => write!(f, "_")?,
            TypeRef::Tuple(elems) => {
                write!(f, "(")?;
                f.write_joined(elems, ", ")?;
                if elems.len() == 1 {
                    write!(f, ",")?;
                }
                write!(f, ")")?;
            }
            TypeRef::Path(path) => path.hir_fmt(f)?,
            TypeRef::RawPtr(inner, mutability) => {
                let mutability = match mutability {
                    hir_def::type_ref::Mutability::Shared => "*const ",
                    hir_def::type_ref::Mutability::Mut => "*mut ",
                };
                write!(f, "{mutability}")?;
                inner.hir_fmt(f)?;
            }
            TypeRef::Reference(inner, lifetime, mutability) => {
                let mutability = match mutability {
                    hir_def::type_ref::Mutability::Shared => "",
                    hir_def::type_ref::Mutability::Mut => "mut ",
                };
                write!(f, "&")?;
                if let Some(lifetime) = lifetime {
                    write!(f, "{} ", lifetime.name)?;
                }
                write!(f, "{mutability}")?;
                inner.hir_fmt(f)?;
            }
            TypeRef::Array(inner, len) => {
                write!(f, "[")?;
                inner.hir_fmt(f)?;
                write!(f, "; {len}]")?;
            }
            TypeRef::Slice(inner) => {
                write!(f, "[")?;
                inner.hir_fmt(f)?;
                write!(f, "]")?;
            }
            &TypeRef::Fn(ref parameters, is_varargs, is_unsafe) => {
                // FIXME: Function pointer qualifiers.
                if is_unsafe {
                    write!(f, "unsafe ")?;
                }
                write!(f, "fn(")?;
                if let Some(((_, return_type), function_parameters)) = parameters.split_last() {
                    for index in 0..function_parameters.len() {
                        let (param_name, param_type) = &function_parameters[index];
                        if let Some(name) = param_name {
                            write!(f, "{name}: ")?;
                        }

                        param_type.hir_fmt(f)?;

                        if index != function_parameters.len() - 1 {
                            write!(f, ", ")?;
                        }
                    }
                    if is_varargs {
                        write!(f, "{}...", if parameters.len() == 1 { "" } else { ", " })?;
                    }
                    write!(f, ")")?;
                    match &return_type {
                        TypeRef::Tuple(tup) if tup.is_empty() => {}
                        _ => {
                            write!(f, " -> ")?;
                            return_type.hir_fmt(f)?;
                        }
                    }
                }
            }
            TypeRef::ImplTrait(bounds) => {
                write!(f, "impl ")?;
                f.write_joined(bounds, " + ")?;
            }
            TypeRef::DynTrait(bounds) => {
                write!(f, "dyn ")?;
                f.write_joined(bounds, " + ")?;
            }
            TypeRef::Macro(macro_call) => {
                let macro_call = macro_call.to_node(f.db.upcast());
                let ctx = body::LowerCtx::with_hygiene(f.db.upcast(), &Hygiene::new_unhygienic());
                match macro_call.path() {
                    Some(path) => match Path::from_src(path, &ctx) {
                        Some(path) => path.hir_fmt(f)?,
                        None => write!(f, "{{macro}}")?,
                    },
                    None => write!(f, "{{macro}}")?,
                }
                write!(f, "!(..)")?;
            }
            TypeRef::Error => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for TypeBound {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            TypeBound::Path(path, modifier) => {
                match modifier {
                    TraitBoundModifier::None => (),
                    TraitBoundModifier::Maybe => write!(f, "?")?,
                }
                path.hir_fmt(f)
            }
            TypeBound::Lifetime(lifetime) => write!(f, "{}", lifetime.name),
            TypeBound::ForLifetime(lifetimes, path) => {
                write!(f, "for<{}> ", lifetimes.iter().format(", "))?;
                path.hir_fmt(f)
            }
            TypeBound::Error => write!(f, "{{error}}"),
        }
    }
}

impl HirDisplay for Path {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match (self.type_anchor(), self.kind()) {
            (Some(anchor), _) => {
                write!(f, "<")?;
                anchor.hir_fmt(f)?;
                write!(f, ">")?;
            }
            (_, PathKind::Plain) => {}
            (_, PathKind::Abs) => {}
            (_, PathKind::Crate) => write!(f, "crate")?,
            (_, PathKind::Super(0)) => write!(f, "self")?,
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
                let crate_graph = f.db.crate_graph();
                let name = crate_graph[*id]
                    .display_name
                    .as_ref()
                    .map(|name| name.canonical_name())
                    .unwrap_or("$crate");
                write!(f, "{name}")?
            }
        }

        for (seg_idx, segment) in self.segments().iter().enumerate() {
            if !matches!(self.kind(), PathKind::Plain) || seg_idx > 0 {
                write!(f, "::")?;
            }
            write!(f, "{}", segment.name)?;
            if let Some(generic_args) = segment.args_and_bindings {
                // We should be in type context, so format as `Foo<Bar>` instead of `Foo::<Bar>`.
                // Do we actually format expressions?
                if generic_args.desugared_from_fn {
                    // First argument will be a tuple, which already includes the parentheses.
                    // If the tuple only contains 1 item, write it manually to avoid the trailing `,`.
                    if let hir_def::path::GenericArg::Type(TypeRef::Tuple(v)) =
                        &generic_args.args[0]
                    {
                        if v.len() == 1 {
                            write!(f, "(")?;
                            v[0].hir_fmt(f)?;
                            write!(f, ")")?;
                        } else {
                            generic_args.args[0].hir_fmt(f)?;
                        }
                    }
                    if let Some(ret) = &generic_args.bindings[0].type_ref {
                        if !matches!(ret, TypeRef::Tuple(v) if v.is_empty()) {
                            write!(f, " -> ")?;
                            ret.hir_fmt(f)?;
                        }
                    }
                    return Ok(());
                }

                write!(f, "<")?;
                let mut first = true;
                for arg in generic_args.args.iter() {
                    if first {
                        first = false;
                        if generic_args.has_self_type {
                            // FIXME: Convert to `<Ty as Trait>` form.
                            write!(f, "Self = ")?;
                        }
                    } else {
                        write!(f, ", ")?;
                    }
                    arg.hir_fmt(f)?;
                }
                for binding in generic_args.bindings.iter() {
                    if first {
                        first = false;
                    } else {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", binding.name)?;
                    match &binding.type_ref {
                        Some(ty) => {
                            write!(f, " = ")?;
                            ty.hir_fmt(f)?
                        }
                        None => {
                            write!(f, ": ")?;
                            f.write_joined(binding.bounds.iter(), " + ")?;
                        }
                    }
                }
                write!(f, ">")?;
            }
        }
        Ok(())
    }
}

impl HirDisplay for hir_def::path::GenericArg {
    fn hir_fmt(&self, f: &mut HirFormatter<'_>) -> Result<(), HirDisplayError> {
        match self {
            hir_def::path::GenericArg::Type(ty) => ty.hir_fmt(f),
            hir_def::path::GenericArg::Const(c) => write!(f, "{c}"),
            hir_def::path::GenericArg::Lifetime(lifetime) => write!(f, "{}", lifetime.name),
        }
    }
}

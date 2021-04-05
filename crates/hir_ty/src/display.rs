//! FIXME: write short doc here

use std::{
    array,
    fmt::{self, Debug},
};

use hir_def::{
    db::DefDatabase,
    find_path,
    generics::TypeParamProvenance,
    item_scope::ItemInNs,
    path::{Path, PathKind},
    type_ref::{TypeBound, TypeRef},
    visibility::Visibility,
    AssocContainerId, Lookup, ModuleId, TraitId,
};
use hir_expand::name::Name;

use crate::{
    db::HirDatabase, from_assoc_type_id, from_foreign_def_id, from_placeholder_idx,
    lt_from_placeholder_idx, primitive, subst_prefix, to_assoc_type_id, traits::chalk::from_chalk,
    utils::generics, AdtId, AliasEq, AliasTy, CallableDefId, CallableSig, DomainGoal, GenericArg,
    ImplTraitId, Interner, Lifetime, LifetimeData, LifetimeOutlives, Mutability, OpaqueTy,
    ProjectionTy, ProjectionTyExt, QuantifiedWhereClause, Scalar, TraitRef, Ty, TyExt, TyKind,
    WhereClause,
};

pub struct HirFormatter<'a> {
    pub db: &'a dyn HirDatabase,
    fmt: &'a mut dyn fmt::Write,
    buf: String,
    curr_size: usize,
    pub(crate) max_size: Option<usize>,
    omit_verbose_types: bool,
    display_target: DisplayTarget,
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError>;

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
                write!(self, "{}", sep)?;
            }
            first = false;
            e.hir_fmt(self)?;
        }
        Ok(())
    }

    /// This allows using the `write!` macro directly with a `HirFormatter`.
    pub fn write_fmt(&mut self, args: fmt::Arguments) -> Result<(), HirDisplayError> {
        // We write to a buffer first to track output size
        self.buf.clear();
        fmt::write(&mut self.buf, args)?;
        self.curr_size += self.buf.len();

        // Then we write to the internal formatter from the buffer
        self.fmt.write_str(&self.buf).map_err(HirDisplayError::from)
    }

    pub fn should_truncate(&self) -> bool {
        if let Some(max_size) = self.max_size {
            self.curr_size >= max_size
        } else {
            false
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

impl<'a, T> fmt::Display for HirDisplayWrapper<'a, T>
where
    T: HirDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.t.hir_fmt(&mut HirFormatter {
            db: self.db,
            fmt: f,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: self.max_size,
            omit_verbose_types: self.omit_verbose_types,
            display_target: self.display_target,
        }) {
            Ok(()) => Ok(()),
            Err(HirDisplayError::FmtError) => Err(fmt::Error),
            Err(HirDisplayError::DisplaySourceCodeError(_)) => {
                // This should never happen
                panic!("HirDisplay failed when calling Display::fmt!")
            }
        }
    }
}

const TYPE_HINT_TRUNCATION: &str = "…";

impl<T: HirDisplay> HirDisplay for &'_ T {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for ProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        let trait_ = f.db.trait_data(self.trait_(f.db));
        let first_parameter = self.self_type_parameter(&Interner).into_displayable(
            f.db,
            f.max_size,
            f.omit_verbose_types,
            f.display_target,
        );
        write!(f, "<{} as {}", first_parameter, trait_.name)?;
        if self.substitution.len(&Interner) > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substitution.interned()[1..], ", ")?;
            write!(f, ">")?;
        }
        write!(f, ">::{}", f.db.type_alias_data(from_assoc_type_id(self.associated_ty_id)).name)?;
        Ok(())
    }
}

impl HirDisplay for OpaqueTy {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        self.substitution.at(&Interner, 0).hir_fmt(f)
    }
}

impl HirDisplay for GenericArg {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self.interned() {
            crate::GenericArgData::Ty(ty) => ty.hir_fmt(f),
        }
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self.kind(&Interner) {
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
            TyKind::Array(t) => {
                write!(f, "[")?;
                t.hir_fmt(f)?;
                write!(f, "; _]")?;
            }
            TyKind::Raw(m, t) | TyKind::Ref(m, t) => {
                let ty_display =
                    t.into_displayable(f.db, f.max_size, f.omit_verbose_types, f.display_target);

                if matches!(self.kind(&Interner), TyKind::Raw(..)) {
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
                let datas;
                let predicates: Vec<_> = match t.kind(&Interner) {
                    TyKind::Dyn(dyn_ty) if dyn_ty.bounds.skip_binders().interned().len() > 1 => {
                        dyn_ty.bounds.skip_binders().interned().iter().cloned().collect()
                    }
                    &TyKind::Alias(AliasTy::Opaque(OpaqueTy {
                        opaque_ty_id,
                        substitution: ref parameters,
                    })) => {
                        let impl_trait_id = f.db.lookup_intern_impl_trait_id(opaque_ty_id.into());
                        if let ImplTraitId::ReturnTypeImplTrait(func, idx) = impl_trait_id {
                            datas =
                                f.db.return_type_impl_traits(func)
                                    .expect("impl trait id without data");
                            let data = (*datas)
                                .as_ref()
                                .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                            let bounds = data.substitute(&Interner, parameters);
                            bounds.into_value_and_skipped_binders().0
                        } else {
                            Vec::new()
                        }
                    }
                    _ => Vec::new(),
                };

                if let Some(WhereClause::Implemented(trait_ref)) =
                    predicates.get(0).map(|b| b.skip_binders())
                {
                    let trait_ = trait_ref.hir_trait_id();
                    if fn_traits(f.db.upcast(), trait_).any(|it| it == trait_)
                        && predicates.len() <= 2
                    {
                        return write!(f, "{}", ty_display);
                    }
                }

                if predicates.len() > 1 {
                    write!(f, "(")?;
                    write!(f, "{}", ty_display)?;
                    write!(f, ")")?;
                } else {
                    write!(f, "{}", ty_display)?;
                }
            }
            TyKind::Tuple(_, substs) => {
                if substs.len(&Interner) == 1 {
                    write!(f, "(")?;
                    substs.at(&Interner, 0).hir_fmt(f)?;
                    write!(f, ",)")?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(&*substs.interned(), ", ")?;
                    write!(f, ")")?;
                }
            }
            TyKind::Function(fn_ptr) => {
                let sig = CallableSig::from_fn_ptr(fn_ptr);
                sig.hir_fmt(f)?;
            }
            TyKind::FnDef(def, parameters) => {
                let def = from_chalk(f.db, *def);
                let sig = f.db.callable_item_signature(def).substitute(&Interner, parameters);
                match def {
                    CallableDefId::FunctionId(ff) => {
                        write!(f, "fn {}", f.db.function_data(ff).name)?
                    }
                    CallableDefId::StructId(s) => write!(f, "{}", f.db.struct_data(s).name)?,
                    CallableDefId::EnumVariantId(e) => {
                        write!(f, "{}", f.db.enum_data(e.parent).variants[e.local_id].name)?
                    }
                };
                if parameters.len(&Interner) > 0 {
                    let generics = generics(f.db.upcast(), def.into());
                    let (parent_params, self_param, type_params, _impl_trait_params) =
                        generics.provenance_split();
                    let total_len = parent_params + self_param + type_params;
                    // We print all params except implicit impl Trait params. Still a bit weird; should we leave out parent and self?
                    if total_len > 0 {
                        write!(f, "<")?;
                        f.write_joined(&parameters.interned()[..total_len], ", ")?;
                        write!(f, ">")?;
                    }
                }
                write!(f, "(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ")")?;
                let ret = sig.ret();
                if !ret.is_unit() {
                    let ret_display = ret.into_displayable(
                        f.db,
                        f.max_size,
                        f.omit_verbose_types,
                        f.display_target,
                    );

                    write!(f, " -> {}", ret_display)?;
                }
            }
            TyKind::Adt(AdtId(def_id), parameters) => {
                match f.display_target {
                    DisplayTarget::Diagnostics | DisplayTarget::Test => {
                        let name = match *def_id {
                            hir_def::AdtId::StructId(it) => f.db.struct_data(it).name.clone(),
                            hir_def::AdtId::UnionId(it) => f.db.union_data(it).name.clone(),
                            hir_def::AdtId::EnumId(it) => f.db.enum_data(it).name.clone(),
                        };
                        write!(f, "{}", name)?;
                    }
                    DisplayTarget::SourceCode { module_id } => {
                        if let Some(path) = find_path::find_path(
                            f.db.upcast(),
                            ItemInNs::Types((*def_id).into()),
                            module_id,
                        ) {
                            write!(f, "{}", path)?;
                        } else {
                            return Err(HirDisplayError::DisplaySourceCodeError(
                                DisplaySourceCodeError::PathNotFound,
                            ));
                        }
                    }
                }

                if parameters.len(&Interner) > 0 {
                    let parameters_to_write = if f.display_target.is_source_code()
                        || f.omit_verbose_types()
                    {
                        match self
                            .as_generic_def(f.db)
                            .map(|generic_def_id| f.db.generic_defaults(generic_def_id))
                            .filter(|defaults| !defaults.is_empty())
                        {
                            None => parameters.interned().as_ref(),
                            Some(default_parameters) => {
                                let mut default_from = 0;
                                for (i, parameter) in parameters.iter(&Interner).enumerate() {
                                    match (
                                        parameter.assert_ty_ref(&Interner).kind(&Interner),
                                        default_parameters.get(i),
                                    ) {
                                        (&TyKind::Error, _) | (_, None) => {
                                            default_from = i + 1;
                                        }
                                        (_, Some(default_parameter)) => {
                                            let actual_default =
                                                default_parameter.clone().substitute(
                                                    &Interner,
                                                    &subst_prefix(parameters, i),
                                                );
                                            if parameter.assert_ty_ref(&Interner) != &actual_default
                                            {
                                                default_from = i + 1;
                                            }
                                        }
                                    }
                                }
                                &parameters.interned()[0..default_from]
                            }
                        }
                    } else {
                        parameters.interned().as_ref()
                    };
                    if !parameters_to_write.is_empty() {
                        write!(f, "<")?;
                        f.write_joined(parameters_to_write, ", ")?;
                        write!(f, ">")?;
                    }
                }
            }
            TyKind::AssociatedType(assoc_type_id, parameters) => {
                let type_alias = from_assoc_type_id(*assoc_type_id);
                let trait_ = match type_alias.lookup(f.db.upcast()).container {
                    AssocContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_ = f.db.trait_data(trait_);
                let type_alias_data = f.db.type_alias_data(type_alias);

                // Use placeholder associated types when the target is test (https://rust-lang.github.io/chalk/book/clauses/type_equality.html#placeholder-associated-types)
                if f.display_target.is_test() {
                    write!(f, "{}::{}", trait_.name, type_alias_data.name)?;
                    if parameters.len(&Interner) > 0 {
                        write!(f, "<")?;
                        f.write_joined(&*parameters.interned(), ", ")?;
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
                let type_alias = f.db.type_alias_data(from_foreign_def_id(*type_alias));
                write!(f, "{}", type_alias.name)?;
            }
            TyKind::OpaqueType(opaque_ty_id, parameters) => {
                let impl_trait_id = f.db.lookup_intern_impl_trait_id((*opaque_ty_id).into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            f.db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data = (*datas)
                            .as_ref()
                            .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                        let bounds = data.substitute(&Interner, &parameters);
                        write_bounds_like_dyn_trait_with_prefix("impl", bounds.skip_binders(), f)?;
                        // FIXME: it would maybe be good to distinguish this from the alias type (when debug printing), and to show the substitution
                    }
                    ImplTraitId::AsyncBlockTypeImplTrait(..) => {
                        write!(f, "impl Future<Output = ")?;
                        parameters.at(&Interner, 0).hir_fmt(f)?;
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
                let sig = substs.at(&Interner, 0).assert_ty_ref(&Interner).callable_sig(f.db);
                if let Some(sig) = sig {
                    if sig.params().is_empty() {
                        write!(f, "||")?;
                    } else if f.omit_verbose_types() {
                        write!(f, "|{}|", TYPE_HINT_TRUNCATION)?;
                    } else {
                        write!(f, "|")?;
                        f.write_joined(sig.params(), ", ")?;
                        write!(f, "|")?;
                    };

                    let ret_display = sig.ret().into_displayable(
                        f.db,
                        f.max_size,
                        f.omit_verbose_types,
                        f.display_target,
                    );
                    write!(f, " -> {}", ret_display)?;
                } else {
                    write!(f, "{{closure}}")?;
                }
            }
            TyKind::Placeholder(idx) => {
                let id = from_placeholder_idx(f.db, *idx);
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.types[id.local_id];
                match param_data.provenance {
                    TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                        write!(f, "{}", param_data.name.clone().unwrap_or_else(Name::missing))?
                    }
                    TypeParamProvenance::ArgumentImplTrait => {
                        let substs = generics.type_params_subst(f.db);
                        let bounds =
                            f.db.generic_predicates(id.parent)
                                .into_iter()
                                .map(|pred| pred.clone().substitute(&Interner, &substs))
                                .filter(|wc| match &wc.skip_binders() {
                                    WhereClause::Implemented(tr) => {
                                        tr.self_type_parameter(&Interner) == self
                                    }
                                    WhereClause::AliasEq(AliasEq {
                                        alias: AliasTy::Projection(proj),
                                        ty: _,
                                    }) => proj.self_type_parameter(&Interner) == self,
                                    _ => false,
                                })
                                .collect::<Vec<_>>();
                        write_bounds_like_dyn_trait_with_prefix("impl", &bounds, f)?;
                    }
                }
            }
            TyKind::BoundVar(idx) => write!(f, "?{}.{}", idx.debruijn.depth(), idx.index)?,
            TyKind::Dyn(dyn_ty) => {
                write_bounds_like_dyn_trait_with_prefix(
                    "dyn",
                    dyn_ty.bounds.skip_binders().interned(),
                    f,
                )?;
            }
            TyKind::Alias(AliasTy::Projection(p_ty)) => p_ty.hir_fmt(f)?,
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                let impl_trait_id = f.db.lookup_intern_impl_trait_id(opaque_ty.opaque_ty_id.into());
                match impl_trait_id {
                    ImplTraitId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            f.db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data = (*datas)
                            .as_ref()
                            .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                        let bounds = data.substitute(&Interner, &opaque_ty.substitution);
                        write_bounds_like_dyn_trait_with_prefix("impl", bounds.skip_binders(), f)?;
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
        }
        Ok(())
    }
}

impl HirDisplay for CallableSig {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
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
            let ret_display =
                ret.into_displayable(f.db, f.max_size, f.omit_verbose_types, f.display_target);
            write!(f, " -> {}", ret_display)?;
        }
        Ok(())
    }
}

fn fn_traits(db: &dyn DefDatabase, trait_: TraitId) -> impl Iterator<Item = TraitId> {
    let krate = trait_.lookup(db).container.krate();
    let fn_traits = [
        db.lang_item(krate, "fn".into()),
        db.lang_item(krate, "fn_mut".into()),
        db.lang_item(krate, "fn_once".into()),
    ];
    array::IntoIter::new(fn_traits).into_iter().flatten().flat_map(|it| it.as_trait())
}

pub fn write_bounds_like_dyn_trait_with_prefix(
    prefix: &str,
    predicates: &[QuantifiedWhereClause],
    f: &mut HirFormatter,
) -> Result<(), HirDisplayError> {
    write!(f, "{}", prefix)?;
    if !predicates.is_empty() {
        write!(f, " ")?;
        write_bounds_like_dyn_trait(predicates, f)
    } else {
        Ok(())
    }
}

fn write_bounds_like_dyn_trait(
    predicates: &[QuantifiedWhereClause],
    f: &mut HirFormatter,
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
    for p in predicates.iter() {
        match p.skip_binders() {
            WhereClause::Implemented(trait_ref) => {
                let trait_ = trait_ref.hir_trait_id();
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
                // We assume that the self type is $0 (i.e. the
                // existential) here, which is the only thing that's
                // possible in actual Rust, and hence don't print it
                write!(f, "{}", f.db.trait_data(trait_).name)?;
                if let [_, params @ ..] = &*trait_ref.substitution.interned() {
                    if is_fn_trait {
                        if let Some(args) =
                            params.first().and_then(|it| it.assert_ty_ref(&Interner).as_tuple())
                        {
                            write!(f, "(")?;
                            f.write_joined(&*args.interned(), ", ")?;
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
                write!(f, " -> ")?;
                alias_eq.ty.hir_fmt(f)?;
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
                    let type_alias =
                        f.db.type_alias_data(from_assoc_type_id(proj.associated_ty_id));
                    write!(f, "{} = ", type_alias.name)?;
                }
                ty.hir_fmt(f)?;
            }
        }
        first = false;
    }
    if angle_open {
        write!(f, ">")?;
    }
    Ok(())
}

impl TraitRef {
    fn hir_fmt_ext(&self, f: &mut HirFormatter, use_as: bool) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        self.self_type_parameter(&Interner).hir_fmt(f)?;
        if use_as {
            write!(f, " as ")?;
        } else {
            write!(f, ": ")?;
        }
        write!(f, "{}", f.db.trait_data(self.hir_trait_id()).name)?;
        if self.substitution.len(&Interner) > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substitution.interned()[1..], ", ")?;
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        self.hir_fmt_ext(f, false)
    }
}

impl HirDisplay for WhereClause {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self {
            WhereClause::Implemented(trait_ref) => trait_ref.hir_fmt(f)?,
            WhereClause::AliasEq(AliasEq { alias: AliasTy::Projection(projection_ty), ty }) => {
                write!(f, "<")?;
                projection_ty.trait_ref(f.db).hir_fmt_ext(f, true)?;
                write!(
                    f,
                    ">::{} = ",
                    f.db.type_alias_data(from_assoc_type_id(projection_ty.associated_ty_id)).name,
                )?;
                ty.hir_fmt(f)?;
            }
            WhereClause::AliasEq(_) => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for LifetimeOutlives {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        self.a.hir_fmt(f)?;
        write!(f, ": ")?;
        self.b.hir_fmt(f)
    }
}

impl HirDisplay for Lifetime {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        self.interned().hir_fmt(f)
    }
}

impl HirDisplay for LifetimeData {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            LifetimeData::BoundVar(idx) => write!(f, "?{}.{}", idx.debruijn.depth(), idx.index),
            LifetimeData::InferenceVar(_) => write!(f, "_"),
            LifetimeData::Placeholder(idx) => {
                let id = lt_from_placeholder_idx(f.db, *idx);
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.lifetimes[id.local_id];
                write!(f, "{}", param_data.name)
            }
            LifetimeData::Static => write!(f, "'static"),
            LifetimeData::Empty(_) => Ok(()),
            LifetimeData::Erased => Ok(()),
            LifetimeData::Phantom(_, _) => Ok(()),
        }
    }
}

impl HirDisplay for DomainGoal {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            DomainGoal::Holds(wc) => {
                write!(f, "Holds(")?;
                wc.hir_fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

pub fn write_visibility(
    module_id: ModuleId,
    vis: Visibility,
    f: &mut HirFormatter,
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
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
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
                write!(f, "{}", mutability)?;
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
                write!(f, "{}", mutability)?;
                inner.hir_fmt(f)?;
            }
            TypeRef::Array(inner) => {
                write!(f, "[")?;
                inner.hir_fmt(f)?;
                // FIXME: Array length?
                write!(f, "; _]")?;
            }
            TypeRef::Slice(inner) => {
                write!(f, "[")?;
                inner.hir_fmt(f)?;
                write!(f, "]")?;
            }
            TypeRef::Fn(tys, is_varargs) => {
                // FIXME: Function pointer qualifiers.
                write!(f, "fn(")?;
                f.write_joined(&tys[..tys.len() - 1], ", ")?;
                if *is_varargs {
                    write!(f, "{}...", if tys.len() == 1 { "" } else { ", " })?;
                }
                write!(f, ")")?;
                let ret_ty = tys.last().unwrap();
                match ret_ty {
                    TypeRef::Tuple(tup) if tup.is_empty() => {}
                    _ => {
                        write!(f, " -> ")?;
                        ret_ty.hir_fmt(f)?;
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
            TypeRef::Error => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for TypeBound {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            TypeBound::Path(path) => path.hir_fmt(f),
            TypeBound::Lifetime(lifetime) => write!(f, "{}", lifetime.name),
            TypeBound::Error => write!(f, "{{error}}"),
        }
    }
}

impl HirDisplay for Path {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match (self.type_anchor(), self.kind()) {
            (Some(anchor), _) => {
                write!(f, "<")?;
                anchor.hir_fmt(f)?;
                write!(f, ">")?;
            }
            (_, PathKind::Plain) => {}
            (_, PathKind::Abs) => write!(f, "::")?,
            (_, PathKind::Crate) => write!(f, "crate")?,
            (_, PathKind::Super(0)) => write!(f, "self")?,
            (_, PathKind::Super(n)) => {
                write!(f, "super")?;
                for _ in 0..*n {
                    write!(f, "::super")?;
                }
            }
            (_, PathKind::DollarCrate(_)) => write!(f, "{{extern_crate}}")?,
        }

        for (seg_idx, segment) in self.segments().iter().enumerate() {
            if seg_idx != 0 {
                write!(f, "::")?;
            }
            write!(f, "{}", segment.name)?;
            if let Some(generic_args) = segment.args_and_bindings {
                // We should be in type context, so format as `Foo<Bar>` instead of `Foo::<Bar>`.
                // Do we actually format expressions?
                write!(f, "<")?;
                let mut first = true;
                for arg in &generic_args.args {
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
                for binding in &generic_args.bindings {
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
                            f.write_joined(&binding.bounds, " + ")?;
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
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            hir_def::path::GenericArg::Type(ty) => ty.hir_fmt(f),
            hir_def::path::GenericArg::Lifetime(lifetime) => write!(f, "{}", lifetime.name),
        }
    }
}

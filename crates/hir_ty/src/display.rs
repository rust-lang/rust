//! FIXME: write short doc here

use std::{borrow::Cow, fmt};

use arrayvec::ArrayVec;
use chalk_ir::Mutability;
use hir_def::{
    db::DefDatabase, find_path, generics::TypeParamProvenance, item_scope::ItemInNs,
    AssocContainerId, Lookup, ModuleId, TraitId,
};
use hir_expand::name::Name;

use crate::{
    db::HirDatabase, from_foreign_def_id, primitive, utils::generics, AdtId, AliasTy,
    CallableDefId, CallableSig, GenericPredicate, Interner, Lifetime, Obligation, OpaqueTy,
    OpaqueTyId, ProjectionTy, Scalar, Substs, TraitRef, Ty, TyKind,
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

impl HirDisplay for &Ty {
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
        let first_parameter = self.parameters[0].into_displayable(
            f.db,
            f.max_size,
            f.omit_verbose_types,
            f.display_target,
        );
        write!(f, "<{} as {}", first_parameter, trait_.name)?;
        if self.parameters.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.parameters[1..], ", ")?;
            write!(f, ">")?;
        }
        write!(f, ">::{}", f.db.type_alias_data(self.associated_ty).name)?;
        Ok(())
    }
}

impl HirDisplay for Ty {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self.interned(&Interner) {
            TyKind::Never => write!(f, "!")?,
            TyKind::Str => write!(f, "str")?,
            TyKind::Scalar(Scalar::Bool) => write!(f, "bool")?,
            TyKind::Scalar(Scalar::Char) => write!(f, "char")?,
            &TyKind::Scalar(Scalar::Float(t)) => write!(f, "{}", primitive::float_ty_to_string(t))?,
            &TyKind::Scalar(Scalar::Int(t)) => write!(f, "{}", primitive::int_ty_to_string(t))?,
            &TyKind::Scalar(Scalar::Uint(t)) => write!(f, "{}", primitive::uint_ty_to_string(t))?,
            TyKind::Slice(parameters) => {
                let t = parameters.as_single();
                write!(f, "[")?;
                t.hir_fmt(f)?;
                write!(f, "]")?;
            }
            TyKind::Array(parameters) => {
                let t = parameters.as_single();
                write!(f, "[")?;
                t.hir_fmt(f)?;
                write!(f, "; _]")?;
            }
            TyKind::Raw(m, parameters) | TyKind::Ref(m, parameters) => {
                let t = parameters.as_single();
                let ty_display =
                    t.into_displayable(f.db, f.max_size, f.omit_verbose_types, f.display_target);

                if matches!(self.interned(&Interner), TyKind::Raw(..)) {
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

                let datas;
                let predicates = match t.interned(&Interner) {
                    TyKind::Dyn(predicates) if predicates.len() > 1 => {
                        Cow::Borrowed(predicates.as_ref())
                    }
                    &TyKind::Alias(AliasTy::Opaque(OpaqueTy {
                        opaque_ty_id: OpaqueTyId::ReturnTypeImplTrait(func, idx),
                        ref parameters,
                    })) => {
                        datas =
                            f.db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data = (*datas)
                            .as_ref()
                            .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                        let bounds = data.subst(parameters);
                        Cow::Owned(bounds.value)
                    }
                    _ => Cow::Borrowed(&[][..]),
                };

                if let [GenericPredicate::Implemented(trait_ref), _] = predicates.as_ref() {
                    let trait_ = trait_ref.trait_;
                    if fn_traits(f.db.upcast(), trait_).any(|it| it == trait_) {
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
                if substs.len() == 1 {
                    write!(f, "(")?;
                    substs[0].hir_fmt(f)?;
                    write!(f, ",)")?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(&*substs.0, ", ")?;
                    write!(f, ")")?;
                }
            }
            TyKind::Function(fn_ptr) => {
                let sig = CallableSig::from_fn_ptr(fn_ptr);
                sig.hir_fmt(f)?;
            }
            TyKind::FnDef(def, parameters) => {
                let def = *def;
                let sig = f.db.callable_item_signature(def).subst(parameters);
                match def {
                    CallableDefId::FunctionId(ff) => {
                        write!(f, "fn {}", f.db.function_data(ff).name)?
                    }
                    CallableDefId::StructId(s) => write!(f, "{}", f.db.struct_data(s).name)?,
                    CallableDefId::EnumVariantId(e) => {
                        write!(f, "{}", f.db.enum_data(e.parent).variants[e.local_id].name)?
                    }
                };
                if parameters.len() > 0 {
                    let generics = generics(f.db.upcast(), def.into());
                    let (parent_params, self_param, type_params, _impl_trait_params) =
                        generics.provenance_split();
                    let total_len = parent_params + self_param + type_params;
                    // We print all params except implicit impl Trait params. Still a bit weird; should we leave out parent and self?
                    if total_len > 0 {
                        write!(f, "<")?;
                        f.write_joined(&parameters.0[..total_len], ", ")?;
                        write!(f, ">")?;
                    }
                }
                write!(f, "(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ")")?;
                let ret = sig.ret();
                if *ret != Ty::unit() {
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

                if parameters.len() > 0 {
                    let parameters_to_write = if f.display_target.is_source_code()
                        || f.omit_verbose_types()
                    {
                        match self
                            .as_generic_def()
                            .map(|generic_def_id| f.db.generic_defaults(generic_def_id))
                            .filter(|defaults| !defaults.is_empty())
                        {
                            None => parameters.0.as_ref(),
                            Some(default_parameters) => {
                                let mut default_from = 0;
                                for (i, parameter) in parameters.iter().enumerate() {
                                    match (parameter.interned(&Interner), default_parameters.get(i))
                                    {
                                        (&TyKind::Unknown, _) | (_, None) => {
                                            default_from = i + 1;
                                        }
                                        (_, Some(default_parameter)) => {
                                            let actual_default = default_parameter
                                                .clone()
                                                .subst(&parameters.prefix(i));
                                            if parameter != &actual_default {
                                                default_from = i + 1;
                                            }
                                        }
                                    }
                                }
                                &parameters.0[0..default_from]
                            }
                        }
                    } else {
                        parameters.0.as_ref()
                    };
                    if !parameters_to_write.is_empty() {
                        write!(f, "<")?;
                        f.write_joined(parameters_to_write, ", ")?;
                        write!(f, ">")?;
                    }
                }
            }
            TyKind::AssociatedType(type_alias, parameters) => {
                let trait_ = match type_alias.lookup(f.db.upcast()).container {
                    AssocContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_ = f.db.trait_data(trait_);
                let type_alias_data = f.db.type_alias_data(*type_alias);

                // Use placeholder associated types when the target is test (https://rust-lang.github.io/chalk/book/clauses/type_equality.html#placeholder-associated-types)
                if f.display_target.is_test() {
                    write!(f, "{}::{}", trait_.name, type_alias_data.name)?;
                    if parameters.len() > 0 {
                        write!(f, "<")?;
                        f.write_joined(&*parameters.0, ", ")?;
                        write!(f, ">")?;
                    }
                } else {
                    let projection_ty =
                        ProjectionTy { associated_ty: *type_alias, parameters: parameters.clone() };

                    projection_ty.hir_fmt(f)?;
                }
            }
            TyKind::ForeignType(type_alias) => {
                let type_alias = f.db.type_alias_data(from_foreign_def_id(*type_alias));
                write!(f, "{}", type_alias.name)?;
            }
            TyKind::OpaqueType(opaque_ty_id, parameters) => {
                match opaque_ty_id {
                    &OpaqueTyId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            f.db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data = (*datas)
                            .as_ref()
                            .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                        let bounds = data.subst(&parameters);
                        write_bounds_like_dyn_trait_with_prefix("impl", &bounds.value, f)?;
                        // FIXME: it would maybe be good to distinguish this from the alias type (when debug printing), and to show the substitution
                    }
                    OpaqueTyId::AsyncBlockTypeImplTrait(..) => {
                        write!(f, "impl Future<Output = ")?;
                        parameters[0].hir_fmt(f)?;
                        write!(f, ">")?;
                    }
                }
            }
            TyKind::Closure(.., substs) => {
                let sig = substs[0].callable_sig(f.db);
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
            TyKind::Placeholder(id) => {
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.types[id.local_id];
                match param_data.provenance {
                    TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                        write!(f, "{}", param_data.name.clone().unwrap_or_else(Name::missing))?
                    }
                    TypeParamProvenance::ArgumentImplTrait => {
                        let bounds = f.db.generic_predicates_for_param(*id);
                        let substs = Substs::type_params_for_generics(&generics);
                        write_bounds_like_dyn_trait_with_prefix(
                            "impl",
                            &bounds.iter().map(|b| b.clone().subst(&substs)).collect::<Vec<_>>(),
                            f,
                        )?;
                    }
                }
            }
            TyKind::BoundVar(idx) => write!(f, "?{}.{}", idx.debruijn.depth(), idx.index)?,
            TyKind::Dyn(predicates) => {
                write_bounds_like_dyn_trait_with_prefix("dyn", predicates, f)?;
            }
            TyKind::Alias(AliasTy::Projection(p_ty)) => p_ty.hir_fmt(f)?,
            TyKind::Alias(AliasTy::Opaque(opaque_ty)) => {
                match opaque_ty.opaque_ty_id {
                    OpaqueTyId::ReturnTypeImplTrait(func, idx) => {
                        let datas =
                            f.db.return_type_impl_traits(func).expect("impl trait id without data");
                        let data = (*datas)
                            .as_ref()
                            .map(|rpit| rpit.impl_traits[idx as usize].bounds.clone());
                        let bounds = data.subst(&opaque_ty.parameters);
                        write_bounds_like_dyn_trait_with_prefix("impl", &bounds.value, f)?;
                    }
                    OpaqueTyId::AsyncBlockTypeImplTrait(..) => {
                        write!(f, "{{async block}}")?;
                    }
                };
            }
            TyKind::Unknown => {
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
        if *ret != Ty::unit() {
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
    // FIXME: Replace ArrayVec when into_iter is a thing on arrays
    ArrayVec::from(fn_traits).into_iter().flatten().flat_map(|it| it.as_trait())
}

pub fn write_bounds_like_dyn_trait_with_prefix(
    prefix: &str,
    predicates: &[GenericPredicate],
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
    predicates: &[GenericPredicate],
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
        match p {
            GenericPredicate::Implemented(trait_ref) => {
                let trait_ = trait_ref.trait_;
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
                if let [_, params @ ..] = &*trait_ref.substs.0 {
                    if is_fn_trait {
                        if let Some(args) = params.first().and_then(|it| it.as_tuple()) {
                            write!(f, "(")?;
                            f.write_joined(&*args.0, ", ")?;
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
            GenericPredicate::Projection(projection_pred) if is_fn_trait => {
                is_fn_trait = false;
                write!(f, " -> ")?;
                projection_pred.ty.hir_fmt(f)?;
            }
            GenericPredicate::Projection(projection_pred) => {
                // in types in actual Rust, these will always come
                // after the corresponding Implemented predicate
                if angle_open {
                    write!(f, ", ")?;
                } else {
                    write!(f, "<")?;
                    angle_open = true;
                }
                let type_alias = f.db.type_alias_data(projection_pred.projection_ty.associated_ty);
                write!(f, "{} = ", type_alias.name)?;
                projection_pred.ty.hir_fmt(f)?;
            }
            GenericPredicate::Error => {
                if angle_open {
                    // impl Trait<X, {error}>
                    write!(f, ", ")?;
                } else if !first {
                    // impl Trait + {error}
                    write!(f, " + ")?;
                }
                p.hir_fmt(f)?;
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

        self.substs[0].hir_fmt(f)?;
        if use_as {
            write!(f, " as ")?;
        } else {
            write!(f, ": ")?;
        }
        write!(f, "{}", f.db.trait_data(self.trait_).name)?;
        if self.substs.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substs[1..], ", ")?;
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

impl HirDisplay for &GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self {
            GenericPredicate::Implemented(trait_ref) => trait_ref.hir_fmt(f)?,
            GenericPredicate::Projection(projection_pred) => {
                write!(f, "<")?;
                projection_pred.projection_ty.trait_ref(f.db).hir_fmt_ext(f, true)?;
                write!(
                    f,
                    ">::{} = ",
                    f.db.type_alias_data(projection_pred.projection_ty.associated_ty).name,
                )?;
                projection_pred.ty.hir_fmt(f)?;
            }
            GenericPredicate::Error => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for Lifetime {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            Lifetime::Parameter(id) => {
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.lifetimes[id.local_id];
                write!(f, "{}", &param_data.name)
            }
            Lifetime::Static => write!(f, "'static"),
        }
    }
}

impl HirDisplay for Obligation {
    fn hir_fmt(&self, f: &mut HirFormatter) -> Result<(), HirDisplayError> {
        match self {
            Obligation::Trait(tr) => {
                write!(f, "Implements(")?;
                tr.hir_fmt(f)?;
                write!(f, ")")
            }
            Obligation::Projection(proj) => {
                write!(f, "Normalize(")?;
                proj.projection_ty.hir_fmt(f)?;
                write!(f, " => ")?;
                proj.ty.hir_fmt(f)?;
                write!(f, ")")
            }
        }
    }
}

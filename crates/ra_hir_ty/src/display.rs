//! FIXME: write short doc here

use std::fmt;

use crate::{
    db::HirDatabase, utils::generics, ApplicationTy, CallableDef, FnSig, GenericPredicate,
    Obligation, ProjectionTy, Substs, TraitRef, Ty, TypeCtor,
};
use hir_def::{generics::TypeParamProvenance, AdtId, AssocContainerId, Lookup};
use hir_expand::name::Name;

pub struct HirFormatter<'a, 'b> {
    pub db: &'a dyn HirDatabase,
    fmt: &'a mut fmt::Formatter<'b>,
    buf: String,
    curr_size: usize,
    pub(crate) max_size: Option<usize>,
    omit_verbose_types: bool,
}

pub trait HirDisplay {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result;

    fn display<'a>(&'a self, db: &'a dyn HirDatabase) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper(db, self, None, false)
    }

    fn display_truncated<'a>(
        &'a self,
        db: &'a dyn HirDatabase,
        max_size: Option<usize>,
    ) -> HirDisplayWrapper<'a, Self>
    where
        Self: Sized,
    {
        HirDisplayWrapper(db, self, max_size, true)
    }
}

impl<'a, 'b> HirFormatter<'a, 'b> {
    pub fn write_joined<T: HirDisplay>(
        &mut self,
        iter: impl IntoIterator<Item = T>,
        sep: &str,
    ) -> fmt::Result {
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
    pub fn write_fmt(&mut self, args: fmt::Arguments) -> fmt::Result {
        // We write to a buffer first to track output size
        self.buf.clear();
        fmt::write(&mut self.buf, args)?;
        self.curr_size += self.buf.len();

        // Then we write to the internal formatter from the buffer
        self.fmt.write_str(&self.buf)
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

pub struct HirDisplayWrapper<'a, T>(&'a dyn HirDatabase, &'a T, Option<usize>, bool);

impl<'a, T> fmt::Display for HirDisplayWrapper<'a, T>
where
    T: HirDisplay,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.1.hir_fmt(&mut HirFormatter {
            db: self.0,
            fmt: f,
            buf: String::with_capacity(20),
            curr_size: 0,
            max_size: self.2,
            omit_verbose_types: self.3,
        })
    }
}

const TYPE_HINT_TRUNCATION: &str = "…";

impl HirDisplay for &Ty {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for ApplicationTy {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self.ctor {
            TypeCtor::Bool => write!(f, "bool")?,
            TypeCtor::Char => write!(f, "char")?,
            TypeCtor::Int(t) => write!(f, "{}", t)?,
            TypeCtor::Float(t) => write!(f, "{}", t)?,
            TypeCtor::Str => write!(f, "str")?,
            TypeCtor::Slice => {
                let t = self.parameters.as_single();
                write!(f, "[{}]", t.display(f.db))?;
            }
            TypeCtor::Array => {
                let t = self.parameters.as_single();
                write!(f, "[{}; _]", t.display(f.db))?;
            }
            TypeCtor::RawPtr(m) => {
                let t = self.parameters.as_single();
                write!(f, "*{}{}", m.as_keyword_for_ptr(), t.display(f.db))?;
            }
            TypeCtor::Ref(m) => {
                let t = self.parameters.as_single();
                let ty_display = if f.omit_verbose_types() {
                    t.display_truncated(f.db, f.max_size)
                } else {
                    t.display(f.db)
                };
                write!(f, "&{}{}", m.as_keyword_for_ref(), ty_display)?;
            }
            TypeCtor::Never => write!(f, "!")?,
            TypeCtor::Tuple { .. } => {
                let ts = &self.parameters;
                if ts.len() == 1 {
                    write!(f, "({},)", ts[0].display(f.db))?;
                } else {
                    write!(f, "(")?;
                    f.write_joined(&*ts.0, ", ")?;
                    write!(f, ")")?;
                }
            }
            TypeCtor::FnPtr { .. } => {
                let sig = FnSig::from_fn_ptr_substs(&self.parameters);
                write!(f, "fn(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            TypeCtor::FnDef(def) => {
                let sig = f.db.callable_item_signature(def).subst(&self.parameters);
                let name = match def {
                    CallableDef::FunctionId(ff) => f.db.function_data(ff).name.clone(),
                    CallableDef::StructId(s) => f.db.struct_data(s).name.clone(),
                    CallableDef::EnumVariantId(e) => {
                        let enum_data = f.db.enum_data(e.parent);
                        enum_data.variants[e.local_id].name.clone()
                    }
                };
                match def {
                    CallableDef::FunctionId(_) => write!(f, "fn {}", name)?,
                    CallableDef::StructId(_) | CallableDef::EnumVariantId(_) => {
                        write!(f, "{}", name)?
                    }
                }
                if self.parameters.len() > 0 {
                    let generics = generics(f.db.upcast(), def.into());
                    let (parent_params, self_param, type_params, _impl_trait_params) =
                        generics.provenance_split();
                    let total_len = parent_params + self_param + type_params;
                    // We print all params except implicit impl Trait params. Still a bit weird; should we leave out parent and self?
                    if total_len > 0 {
                        write!(f, "<")?;
                        f.write_joined(&self.parameters.0[..total_len], ", ")?;
                        write!(f, ">")?;
                    }
                }
                write!(f, "(")?;
                f.write_joined(sig.params(), ", ")?;
                write!(f, ") -> {}", sig.ret().display(f.db))?;
            }
            TypeCtor::Adt(def_id) => {
                let name = match def_id {
                    AdtId::StructId(it) => f.db.struct_data(it).name.clone(),
                    AdtId::UnionId(it) => f.db.union_data(it).name.clone(),
                    AdtId::EnumId(it) => f.db.enum_data(it).name.clone(),
                };
                write!(f, "{}", name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;

                    let mut non_default_parameters = Vec::with_capacity(self.parameters.len());
                    let parameters_to_write = if f.omit_verbose_types() {
                        match self
                            .ctor
                            .as_generic_def()
                            .map(|generic_def_id| f.db.generic_defaults(generic_def_id))
                            .filter(|defaults| !defaults.is_empty())
                        {
                            Option::None => self.parameters.0.as_ref(),
                            Option::Some(default_parameters) => {
                                for (i, parameter) in self.parameters.iter().enumerate() {
                                    match (parameter, default_parameters.get(i)) {
                                        (&Ty::Unknown, _) | (_, None) => {
                                            non_default_parameters.push(parameter.clone())
                                        }
                                        (_, Some(default_parameter))
                                            if parameter != default_parameter =>
                                        {
                                            non_default_parameters.push(parameter.clone())
                                        }
                                        _ => (),
                                    }
                                }
                                &non_default_parameters
                            }
                        }
                    } else {
                        self.parameters.0.as_ref()
                    };

                    f.write_joined(parameters_to_write, ", ")?;
                    write!(f, ">")?;
                }
            }
            TypeCtor::AssociatedType(type_alias) => {
                let trait_ = match type_alias.lookup(f.db.upcast()).container {
                    AssocContainerId::TraitId(it) => it,
                    _ => panic!("not an associated type"),
                };
                let trait_name = f.db.trait_data(trait_).name.clone();
                let name = f.db.type_alias_data(type_alias).name.clone();
                write!(f, "{}::{}", trait_name, name)?;
                if self.parameters.len() > 0 {
                    write!(f, "<")?;
                    f.write_joined(&*self.parameters.0, ", ")?;
                    write!(f, ">")?;
                }
            }
            TypeCtor::Closure { .. } => {
                let sig = self.parameters[0]
                    .callable_sig(f.db)
                    .expect("first closure parameter should contain signature");
                let return_type_hint = sig.ret().display(f.db);
                if sig.params().is_empty() {
                    write!(f, "|| -> {}", return_type_hint)?;
                } else if f.omit_verbose_types() {
                    write!(f, "|{}| -> {}", TYPE_HINT_TRUNCATION, return_type_hint)?;
                } else {
                    write!(f, "|")?;
                    f.write_joined(sig.params(), ", ")?;
                    write!(f, "| -> {}", return_type_hint)?;
                };
            }
        }
        Ok(())
    }
}

impl HirDisplay for ProjectionTy {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        let trait_name = f.db.trait_data(self.trait_(f.db)).name.clone();
        write!(f, "<{} as {}", self.parameters[0].display(f.db), trait_name,)?;
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
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        match self {
            Ty::Apply(a_ty) => a_ty.hir_fmt(f)?,
            Ty::Projection(p_ty) => p_ty.hir_fmt(f)?,
            Ty::Placeholder(id) => {
                let generics = generics(f.db.upcast(), id.parent);
                let param_data = &generics.params.types[id.local_id];
                match param_data.provenance {
                    TypeParamProvenance::TypeParamList | TypeParamProvenance::TraitSelf => {
                        write!(f, "{}", param_data.name.clone().unwrap_or_else(Name::missing))?
                    }
                    TypeParamProvenance::ArgumentImplTrait => {
                        write!(f, "impl ")?;
                        let bounds = f.db.generic_predicates_for_param(*id);
                        let substs = Substs::type_params_for_generics(&generics);
                        write_bounds_like_dyn_trait(
                            &bounds.iter().map(|b| b.clone().subst(&substs)).collect::<Vec<_>>(),
                            f,
                        )?;
                    }
                }
            }
            Ty::Bound(idx) => write!(f, "?{}", idx)?,
            Ty::Dyn(predicates) | Ty::Opaque(predicates) => {
                match self {
                    Ty::Dyn(_) => write!(f, "dyn ")?,
                    Ty::Opaque(_) => write!(f, "impl ")?,
                    _ => unreachable!(),
                };
                write_bounds_like_dyn_trait(&predicates, f)?;
            }
            Ty::Unknown => write!(f, "{{unknown}}")?,
            Ty::Infer(..) => write!(f, "_")?,
        }
        Ok(())
    }
}

fn write_bounds_like_dyn_trait(
    predicates: &[GenericPredicate],
    f: &mut HirFormatter,
) -> fmt::Result {
    // Note: This code is written to produce nice results (i.e.
    // corresponding to surface Rust) for types that can occur in
    // actual Rust. It will have weird results if the predicates
    // aren't as expected (i.e. self types = $0, projection
    // predicates for a certain trait come after the Implemented
    // predicate for that trait).
    let mut first = true;
    let mut angle_open = false;
    for p in predicates.iter() {
        match p {
            GenericPredicate::Implemented(trait_ref) => {
                if angle_open {
                    write!(f, ">")?;
                }
                if !first {
                    write!(f, " + ")?;
                }
                // We assume that the self type is $0 (i.e. the
                // existential) here, which is the only thing that's
                // possible in actual Rust, and hence don't print it
                write!(f, "{}", f.db.trait_data(trait_ref.trait_).name.clone())?;
                if trait_ref.substs.len() > 1 {
                    write!(f, "<")?;
                    f.write_joined(&trait_ref.substs[1..], ", ")?;
                    // there might be assoc type bindings, so we leave the angle brackets open
                    angle_open = true;
                }
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
                let name =
                    f.db.type_alias_data(projection_pred.projection_ty.associated_ty).name.clone();
                write!(f, "{} = ", name)?;
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
    fn hir_fmt_ext(&self, f: &mut HirFormatter, use_as: bool) -> fmt::Result {
        if f.should_truncate() {
            return write!(f, "{}", TYPE_HINT_TRUNCATION);
        }

        self.substs[0].hir_fmt(f)?;
        if use_as {
            write!(f, " as ")?;
        } else {
            write!(f, ": ")?;
        }
        write!(f, "{}", f.db.trait_data(self.trait_).name.clone())?;
        if self.substs.len() > 1 {
            write!(f, "<")?;
            f.write_joined(&self.substs[1..], ", ")?;
            write!(f, ">")?;
        }
        Ok(())
    }
}

impl HirDisplay for TraitRef {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        self.hir_fmt_ext(f, false)
    }
}

impl HirDisplay for &GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        HirDisplay::hir_fmt(*self, f)
    }
}

impl HirDisplay for GenericPredicate {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
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
                    ">::{} = {}",
                    f.db.type_alias_data(projection_pred.projection_ty.associated_ty).name,
                    projection_pred.ty.display(f.db)
                )?;
            }
            GenericPredicate::Error => write!(f, "{{error}}")?,
        }
        Ok(())
    }
}

impl HirDisplay for Obligation {
    fn hir_fmt(&self, f: &mut HirFormatter) -> fmt::Result {
        match self {
            Obligation::Trait(tr) => write!(f, "Implements({})", tr.display(f.db)),
            Obligation::Projection(proj) => write!(
                f,
                "Normalize({} => {})",
                proj.projection_ty.display(f.db),
                proj.ty.display(f.db)
            ),
        }
    }
}

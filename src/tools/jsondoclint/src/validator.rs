use std::collections::HashSet;
use std::hash::Hash;

use rustdoc_json_types::{
    Constant, Crate, DynTrait, Enum, FnDecl, Function, FunctionPointer, GenericArg, GenericArgs,
    GenericBound, GenericParamDef, Generics, Id, Impl, Import, ItemEnum, ItemSummary, Module,
    OpaqueTy, Path, Primitive, ProcMacro, Static, Struct, StructKind, Term, Trait, TraitAlias,
    Type, TypeBinding, TypeBindingKind, Typedef, Union, Variant, VariantKind, WherePredicate,
};
use serde_json::Value;

use crate::{item_kind::Kind, json_find, Error, ErrorKind};

// This is a rustc implementation detail that we rely on here
const LOCAL_CRATE_ID: u32 = 0;

/// The Validator walks over the JSON tree, and ensures it is well formed.
/// It is made of several parts.
///
/// - `check_*`: These take a type from [`rustdoc_json_types`], and check that
///              it is well formed. This involves calling `check_*` functions on
///              fields of that item, and `add_*` functions on [`Id`]s.
/// - `add_*`: These add an [`Id`] to the worklist, after validating it to check if
///            the `Id` is a kind expected in this suituation.
#[derive(Debug)]
pub struct Validator<'a> {
    pub(crate) errs: Vec<Error>,
    krate: &'a Crate,
    krate_json: Value,
    /// Worklist of Ids to check.
    todo: HashSet<&'a Id>,
    /// Ids that have already been visited, so don't need to be checked again.
    seen_ids: HashSet<&'a Id>,
    /// Ids that have already been reported missing.
    missing_ids: HashSet<&'a Id>,
}

enum PathKind {
    Trait,
    /// Structs, Enums, Unions and Typedefs.
    ///
    /// This doesn't include trait's because traits are not types.
    Type,
}

impl<'a> Validator<'a> {
    pub fn new(krate: &'a Crate, krate_json: Value) -> Self {
        Self {
            krate,
            krate_json,
            errs: Vec::new(),
            seen_ids: HashSet::new(),
            todo: HashSet::new(),
            missing_ids: HashSet::new(),
        }
    }

    pub fn check_crate(&mut self) {
        // Graph traverse the index
        let root = &self.krate.root;
        self.add_mod_id(root);
        while let Some(id) = set_remove(&mut self.todo) {
            self.seen_ids.insert(id);
            self.check_item(id);
        }

        let root_crate_id = self.krate.index[root].crate_id;
        assert_eq!(root_crate_id, LOCAL_CRATE_ID, "LOCAL_CRATE_ID is wrong");
        for (id, item_info) in &self.krate.paths {
            self.check_item_info(id, item_info);
        }
    }

    fn check_items(&mut self, id: &Id, items: &[Id]) {
        let mut visited_ids = HashSet::with_capacity(items.len());

        for item in items {
            if !visited_ids.insert(item) {
                self.fail(
                    id,
                    ErrorKind::Custom(format!("Duplicated entry in `items` field: `{item:?}`")),
                );
            }
        }
    }

    fn check_item(&mut self, id: &'a Id) {
        if let Some(item) = &self.krate.index.get(id) {
            item.links.values().for_each(|id| self.add_any_id(id));

            match &item.inner {
                ItemEnum::Import(x) => self.check_import(x),
                ItemEnum::Union(x) => self.check_union(x),
                ItemEnum::Struct(x) => self.check_struct(x),
                ItemEnum::StructField(x) => self.check_struct_field(x),
                ItemEnum::Enum(x) => self.check_enum(x),
                ItemEnum::Variant(x) => self.check_variant(x, id),
                ItemEnum::Function(x) => self.check_function(x),
                ItemEnum::Trait(x) => self.check_trait(x, id),
                ItemEnum::TraitAlias(x) => self.check_trait_alias(x),
                ItemEnum::Impl(x) => self.check_impl(x, id),
                ItemEnum::Typedef(x) => self.check_typedef(x),
                ItemEnum::OpaqueTy(x) => self.check_opaque_ty(x),
                ItemEnum::Constant(x) => self.check_constant(x),
                ItemEnum::Static(x) => self.check_static(x),
                ItemEnum::ForeignType => {} // nop
                ItemEnum::Macro(x) => self.check_macro(x),
                ItemEnum::ProcMacro(x) => self.check_proc_macro(x),
                ItemEnum::Primitive(x) => self.check_primitive_type(x),
                ItemEnum::Module(x) => self.check_module(x, id),
                // FIXME: Why don't these have their own structs?
                ItemEnum::ExternCrate { .. } => {}
                ItemEnum::AssocConst { type_, default: _ } => self.check_type(type_),
                ItemEnum::AssocType { generics, bounds, default } => {
                    self.check_generics(generics);
                    bounds.iter().for_each(|b| self.check_generic_bound(b));
                    if let Some(ty) = default {
                        self.check_type(ty);
                    }
                }
            }
        } else {
            assert!(self.krate.paths.contains_key(id));
        }
    }

    // Core checkers
    fn check_module(&mut self, module: &'a Module, id: &Id) {
        self.check_items(id, &module.items);
        module.items.iter().for_each(|i| self.add_mod_item_id(i));
    }

    fn check_import(&mut self, x: &'a Import) {
        if x.glob {
            self.add_glob_import_item_id(x.id.as_ref().unwrap());
        } else if let Some(id) = &x.id {
            self.add_import_item_id(id);
        }
    }

    fn check_union(&mut self, x: &'a Union) {
        self.check_generics(&x.generics);
        x.fields.iter().for_each(|i| self.add_field_id(i));
        x.impls.iter().for_each(|i| self.add_impl_id(i));
    }

    fn check_struct(&mut self, x: &'a Struct) {
        self.check_generics(&x.generics);
        match &x.kind {
            StructKind::Unit => {}
            StructKind::Tuple(fields) => fields.iter().flatten().for_each(|f| self.add_field_id(f)),
            StructKind::Plain { fields, fields_stripped: _ } => {
                fields.iter().for_each(|f| self.add_field_id(f))
            }
        }
        x.impls.iter().for_each(|i| self.add_impl_id(i));
    }

    fn check_struct_field(&mut self, x: &'a Type) {
        self.check_type(x);
    }

    fn check_enum(&mut self, x: &'a Enum) {
        self.check_generics(&x.generics);
        x.variants.iter().for_each(|i| self.add_variant_id(i));
        x.impls.iter().for_each(|i| self.add_impl_id(i));
    }

    fn check_variant(&mut self, x: &'a Variant, id: &'a Id) {
        let Variant { kind, discriminant } = x;

        if let Some(discr) = discriminant {
            if let (Err(_), Err(_)) = (discr.value.parse::<i128>(), discr.value.parse::<u128>()) {
                self.fail(
                    id,
                    ErrorKind::Custom(format!(
                        "Failed to parse discriminant value `{}`",
                        discr.value
                    )),
                );
            }
        }

        match kind {
            VariantKind::Plain => {}
            VariantKind::Tuple(tys) => tys.iter().flatten().for_each(|t| self.add_field_id(t)),
            VariantKind::Struct { fields, fields_stripped: _ } => {
                fields.iter().for_each(|f| self.add_field_id(f))
            }
        }
    }

    fn check_function(&mut self, x: &'a Function) {
        self.check_generics(&x.generics);
        self.check_fn_decl(&x.decl);
    }

    fn check_trait(&mut self, x: &'a Trait, id: &Id) {
        self.check_items(id, &x.items);
        self.check_generics(&x.generics);
        x.items.iter().for_each(|i| self.add_trait_item_id(i));
        x.bounds.iter().for_each(|i| self.check_generic_bound(i));
        x.implementations.iter().for_each(|i| self.add_impl_id(i));
    }

    fn check_trait_alias(&mut self, x: &'a TraitAlias) {
        self.check_generics(&x.generics);
        x.params.iter().for_each(|i| self.check_generic_bound(i));
    }

    fn check_impl(&mut self, x: &'a Impl, id: &Id) {
        self.check_items(id, &x.items);
        self.check_generics(&x.generics);
        if let Some(path) = &x.trait_ {
            self.check_path(path, PathKind::Trait);
        }
        self.check_type(&x.for_);
        x.items.iter().for_each(|i| self.add_trait_item_id(i));
        if let Some(blanket_impl) = &x.blanket_impl {
            self.check_type(blanket_impl)
        }
    }

    fn check_typedef(&mut self, x: &'a Typedef) {
        self.check_generics(&x.generics);
        self.check_type(&x.type_);
    }

    fn check_opaque_ty(&mut self, x: &'a OpaqueTy) {
        x.bounds.iter().for_each(|b| self.check_generic_bound(b));
        self.check_generics(&x.generics);
    }

    fn check_constant(&mut self, x: &'a Constant) {
        self.check_type(&x.type_);
    }

    fn check_static(&mut self, x: &'a Static) {
        self.check_type(&x.type_);
    }

    fn check_macro(&mut self, _: &'a str) {
        // nop
    }

    fn check_proc_macro(&mut self, _: &'a ProcMacro) {
        // nop
    }

    fn check_primitive_type(&mut self, x: &'a Primitive) {
        x.impls.iter().for_each(|i| self.add_impl_id(i));
    }

    fn check_generics(&mut self, x: &'a Generics) {
        x.params.iter().for_each(|p| self.check_generic_param_def(p));
        x.where_predicates.iter().for_each(|w| self.check_where_predicate(w));
    }

    fn check_type(&mut self, x: &'a Type) {
        match x {
            Type::ResolvedPath(path) => self.check_path(path, PathKind::Type),
            Type::DynTrait(dyn_trait) => self.check_dyn_trait(dyn_trait),
            Type::Generic(_) => {}
            Type::Primitive(_) => {}
            Type::FunctionPointer(fp) => self.check_function_pointer(&**fp),
            Type::Tuple(tys) => tys.iter().for_each(|ty| self.check_type(ty)),
            Type::Slice(inner) => self.check_type(&**inner),
            Type::Array { type_, len: _ } => self.check_type(&**type_),
            Type::ImplTrait(bounds) => bounds.iter().for_each(|b| self.check_generic_bound(b)),
            Type::Infer => {}
            Type::RawPointer { mutable: _, type_ } => self.check_type(&**type_),
            Type::BorrowedRef { lifetime: _, mutable: _, type_ } => self.check_type(&**type_),
            Type::QualifiedPath { name: _, args, self_type, trait_ } => {
                self.check_generic_args(&**args);
                self.check_type(&**self_type);
                if let Some(trait_) = trait_ {
                    self.check_path(trait_, PathKind::Trait);
                }
            }
        }
    }

    fn check_fn_decl(&mut self, x: &'a FnDecl) {
        x.inputs.iter().for_each(|(_name, ty)| self.check_type(ty));
        if let Some(output) = &x.output {
            self.check_type(output);
        }
    }

    fn check_generic_bound(&mut self, x: &'a GenericBound) {
        match x {
            GenericBound::TraitBound { trait_, generic_params, modifier: _ } => {
                self.check_path(trait_, PathKind::Trait);
                generic_params.iter().for_each(|gpd| self.check_generic_param_def(gpd));
            }
            GenericBound::Outlives(_) => {}
        }
    }

    fn check_path(&mut self, x: &'a Path, kind: PathKind) {
        match kind {
            PathKind::Trait => self.add_trait_or_alias_id(&x.id),
            PathKind::Type => self.add_type_id(&x.id),
        }
        if let Some(args) = &x.args {
            self.check_generic_args(&**args);
        }
    }

    fn check_generic_args(&mut self, x: &'a GenericArgs) {
        match x {
            GenericArgs::AngleBracketed { args, bindings } => {
                args.iter().for_each(|arg| self.check_generic_arg(arg));
                bindings.iter().for_each(|bind| self.check_type_binding(bind));
            }
            GenericArgs::Parenthesized { inputs, output } => {
                inputs.iter().for_each(|ty| self.check_type(ty));
                if let Some(o) = output {
                    self.check_type(o);
                }
            }
        }
    }

    fn check_generic_param_def(&mut self, gpd: &'a GenericParamDef) {
        match &gpd.kind {
            rustdoc_json_types::GenericParamDefKind::Lifetime { outlives: _ } => {}
            rustdoc_json_types::GenericParamDefKind::Type { bounds, default, synthetic: _ } => {
                bounds.iter().for_each(|b| self.check_generic_bound(b));
                if let Some(ty) = default {
                    self.check_type(ty);
                }
            }
            rustdoc_json_types::GenericParamDefKind::Const { type_, default: _ } => {
                self.check_type(type_)
            }
        }
    }

    fn check_generic_arg(&mut self, arg: &'a GenericArg) {
        match arg {
            GenericArg::Lifetime(_) => {}
            GenericArg::Type(ty) => self.check_type(ty),
            GenericArg::Const(c) => self.check_constant(c),
            GenericArg::Infer => {}
        }
    }

    fn check_type_binding(&mut self, bind: &'a TypeBinding) {
        self.check_generic_args(&bind.args);
        match &bind.binding {
            TypeBindingKind::Equality(term) => self.check_term(term),
            TypeBindingKind::Constraint(bounds) => {
                bounds.iter().for_each(|b| self.check_generic_bound(b))
            }
        }
    }

    fn check_term(&mut self, term: &'a Term) {
        match term {
            Term::Type(ty) => self.check_type(ty),
            Term::Constant(con) => self.check_constant(con),
        }
    }

    fn check_where_predicate(&mut self, w: &'a WherePredicate) {
        match w {
            WherePredicate::BoundPredicate { type_, bounds, generic_params } => {
                self.check_type(type_);
                bounds.iter().for_each(|b| self.check_generic_bound(b));
                generic_params.iter().for_each(|gpd| self.check_generic_param_def(gpd));
            }
            WherePredicate::RegionPredicate { lifetime: _, bounds } => {
                bounds.iter().for_each(|b| self.check_generic_bound(b));
            }
            WherePredicate::EqPredicate { lhs, rhs } => {
                self.check_type(lhs);
                self.check_term(rhs);
            }
        }
    }

    fn check_dyn_trait(&mut self, dyn_trait: &'a DynTrait) {
        for pt in &dyn_trait.traits {
            self.check_path(&pt.trait_, PathKind::Trait);
            pt.generic_params.iter().for_each(|gpd| self.check_generic_param_def(gpd));
        }
    }

    fn check_function_pointer(&mut self, fp: &'a FunctionPointer) {
        self.check_fn_decl(&fp.decl);
        fp.generic_params.iter().for_each(|gpd| self.check_generic_param_def(gpd));
    }

    fn check_item_info(&mut self, id: &Id, item_info: &ItemSummary) {
        // FIXME: Their should be a better way to determine if an item is local, rather than relying on `LOCAL_CRATE_ID`,
        // which encodes rustc implementation details.
        if item_info.crate_id == LOCAL_CRATE_ID && !self.krate.index.contains_key(id) {
            self.errs.push(Error {
                id: id.clone(),
                kind: ErrorKind::Custom(
                    "Id for local item in `paths` but not in `index`".to_owned(),
                ),
            })
        }
    }

    fn add_id_checked(&mut self, id: &'a Id, valid: fn(Kind) -> bool, expected: &str) {
        if let Some(kind) = self.kind_of(id) {
            if valid(kind) {
                if !self.seen_ids.contains(id) {
                    self.todo.insert(id);
                }
            } else {
                self.fail_expecting(id, expected);
            }
        } else {
            if !self.missing_ids.contains(id) {
                self.missing_ids.insert(id);

                let sels = json_find::find_selector(&self.krate_json, &Value::String(id.0.clone()));
                assert_ne!(sels.len(), 0);

                self.fail(id, ErrorKind::NotFound(sels))
            }
        }
    }

    fn add_any_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, |_| true, "any kind of item");
    }

    fn add_field_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_struct_field, "StructField");
    }

    fn add_mod_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_module, "Module");
    }
    fn add_impl_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_impl, "Impl");
    }

    fn add_variant_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_variant, "Variant");
    }

    fn add_trait_or_alias_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_trait_or_alias, "Trait (or TraitAlias)");
    }

    fn add_type_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::is_type, "Type (Struct, Enum, Union or Typedef)");
    }

    /// Add an Id that appeared in a trait
    fn add_trait_item_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::can_appear_in_trait, "Trait inner item");
    }

    /// Add an Id that can be `use`d
    fn add_import_item_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::can_appear_in_import, "Import inner item");
    }

    fn add_glob_import_item_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::can_appear_in_glob_import, "Glob import inner item");
    }

    /// Add an Id that appeared in a mod
    fn add_mod_item_id(&mut self, id: &'a Id) {
        self.add_id_checked(id, Kind::can_appear_in_mod, "Module inner item")
    }

    fn fail_expecting(&mut self, id: &Id, expected: &str) {
        let kind = self.kind_of(id).unwrap(); // We know it has a kind, as it's wrong.
        self.fail(id, ErrorKind::Custom(format!("Expected {expected} but found {kind:?}")));
    }

    fn fail(&mut self, id: &Id, kind: ErrorKind) {
        self.errs.push(Error { id: id.clone(), kind });
    }

    fn kind_of(&mut self, id: &Id) -> Option<Kind> {
        if let Some(item) = self.krate.index.get(id) {
            Some(Kind::from_item(item))
        } else if let Some(summary) = self.krate.paths.get(id) {
            Some(Kind::from_summary(summary))
        } else {
            None
        }
    }
}

fn set_remove<T: Hash + Eq + Clone>(set: &mut HashSet<T>) -> Option<T> {
    if let Some(id) = set.iter().next() {
        let id = id.clone();
        set.take(&id)
    } else {
        None
    }
}

#[cfg(test)]
mod tests;

//! Type tree for term search

use hir_def::find_path::PrefixKind;
use hir_expand::mod_path::ModPath;
use hir_ty::{db::HirDatabase, display::HirDisplay};
use itertools::Itertools;

use crate::{
    Adt, AsAssocItem, Const, ConstParam, Field, Function, GenericDef, Local, ModuleDef,
    SemanticsScope, Static, Struct, StructKind, Trait, Type, Variant,
};

/// Helper function to get path to `ModuleDef`
fn mod_item_path(sema_scope: &SemanticsScope<'_>, def: &ModuleDef) -> Option<ModPath> {
    let db = sema_scope.db;
    // Account for locals shadowing items from module
    let name_hit_count = def.name(db).map(|def_name| {
        let mut name_hit_count = 0;
        sema_scope.process_all_names(&mut |name, _| {
            if name == def_name {
                name_hit_count += 1;
            }
        });
        name_hit_count
    });

    let m = sema_scope.module();
    match name_hit_count {
        Some(0..=1) | None => m.find_use_path(db.upcast(), *def, false, true),
        Some(_) => m.find_use_path_prefixed(db.upcast(), *def, PrefixKind::ByCrate, false, true),
    }
}

/// Helper function to get path to `ModuleDef` as string
fn mod_item_path_str(sema_scope: &SemanticsScope<'_>, def: &ModuleDef) -> String {
    let path = mod_item_path(sema_scope, def);
    path.map(|it| it.display(sema_scope.db.upcast()).to_string()).unwrap()
}

/// Helper function to get path to `Type`
fn type_path(sema_scope: &SemanticsScope<'_>, ty: &Type) -> String {
    let db = sema_scope.db;
    match ty.as_adt() {
        Some(adt) => {
            let ty_name = ty.display(db).to_string();

            let mut path = mod_item_path(sema_scope, &ModuleDef::Adt(adt)).unwrap();
            path.pop_segment();
            let path = path.display(db.upcast()).to_string();
            match path.is_empty() {
                true => ty_name,
                false => format!("{path}::{ty_name}"),
            }
        }
        None => ty.display(db).to_string(),
    }
}

/// Helper function to filter out generic parameters that are default
fn non_default_generics(db: &dyn HirDatabase, def: GenericDef, generics: &[Type]) -> Vec<Type> {
    def.type_params(db)
        .into_iter()
        .zip(generics)
        .filter(|(tp, arg)| tp.default(db).as_ref() != Some(arg))
        .map(|(_, arg)| arg.clone())
        .collect()
}

/// Type tree shows how can we get from set of types to some type.
///
/// Consider the following code as an example
/// ```
/// fn foo(x: i32, y: bool) -> Option<i32> { None }
/// fn bar() {
///    let a = 1;
///    let b = true;
///    let c: Option<i32> = _;
/// }
/// ```
/// If we generate type tree in the place of `_` we get
/// ```txt
///       Option<i32>
///           |
///     foo(i32, bool)
///      /        \
///  a: i32      b: bool
/// ```
/// So in short it pretty much gives us a way to get type `Option<i32>` using the items we have in
/// scope.
#[derive(Debug, Clone, Eq, Hash, PartialEq)]
pub enum TypeTree {
    /// Constant
    Const(Const),
    /// Static variable
    Static(Static),
    /// Local variable
    Local(Local),
    /// Constant generic parameter
    ConstParam(ConstParam),
    /// Well known type (such as `true` for bool)
    FamousType { ty: Type, value: &'static str },
    /// Function or method call
    Function { func: Function, generics: Vec<Type>, params: Vec<TypeTree> },
    /// Enum variant construction
    Variant { variant: Variant, generics: Vec<Type>, params: Vec<TypeTree> },
    /// Struct construction
    Struct { strukt: Struct, generics: Vec<Type>, params: Vec<TypeTree> },
    /// Struct field access
    Field { type_tree: Box<TypeTree>, field: Field },
    /// Passing type as reference (with `&`)
    Reference(Box<TypeTree>),
}

impl TypeTree {
    /// Generate source code for type tree.
    ///
    /// Note that trait imports are not added to generated code.
    /// To make sure that the code is valid, callee has to also ensure that all the traits listed
    /// by `traits_used` method are also imported.
    pub fn gen_source_code(&self, sema_scope: &SemanticsScope<'_>) -> String {
        let db = sema_scope.db;
        match self {
            TypeTree::Const(it) => mod_item_path_str(sema_scope, &ModuleDef::Const(*it)),
            TypeTree::Static(it) => mod_item_path_str(sema_scope, &ModuleDef::Static(*it)),
            TypeTree::Local(it) => return it.name(db).display(db.upcast()).to_string(),
            TypeTree::ConstParam(it) => return it.name(db).display(db.upcast()).to_string(),
            TypeTree::FamousType { value, .. } => return value.to_string(),
            TypeTree::Function { func, params, .. } => {
                if let Some(self_param) = func.self_param(db) {
                    let func_name = func.name(db).display(db.upcast()).to_string();
                    let target = params.first().expect("no self param").gen_source_code(sema_scope);
                    let args =
                        params.iter().skip(1).map(|f| f.gen_source_code(sema_scope)).join(", ");

                    match func.as_assoc_item(db).unwrap().containing_trait_or_trait_impl(db) {
                        Some(trait_) => {
                            let trait_name =
                                mod_item_path_str(sema_scope, &ModuleDef::Trait(trait_));
                            let target = match self_param.access(db) {
                                crate::Access::Shared => format!("&{target}"),
                                crate::Access::Exclusive => format!("&mut {target}"),
                                crate::Access::Owned => target,
                            };
                            match args.is_empty() {
                                true => format!("{trait_name}::{func_name}({target})",),
                                false => format!("{trait_name}::{func_name}({target}, {args})",),
                            }
                        }
                        None => format!("{target}.{func_name}({args})"),
                    }
                } else {
                    let args = params.iter().map(|f| f.gen_source_code(sema_scope)).join(", ");

                    match func.as_assoc_item(db).map(|it| it.container(db)) {
                        Some(container) => {
                            let container_name = match container {
                                crate::AssocItemContainer::Trait(trait_) => {
                                    mod_item_path_str(sema_scope, &ModuleDef::Trait(trait_))
                                }
                                crate::AssocItemContainer::Impl(imp) => {
                                    let self_ty = imp.self_ty(db);
                                    // Should it be guaranteed that `mod_item_path` always exists?
                                    match self_ty
                                        .as_adt()
                                        .and_then(|adt| mod_item_path(sema_scope, &adt.into()))
                                    {
                                        Some(path) => {
                                            path.display(sema_scope.db.upcast()).to_string()
                                        }
                                        None => self_ty.display(db).to_string(),
                                    }
                                }
                            };
                            let fn_name = func.name(db).display(db.upcast()).to_string();
                            format!("{container_name}::{fn_name}({args})",)
                        }
                        None => {
                            let fn_name =
                                mod_item_path_str(sema_scope, &ModuleDef::Function(*func));
                            format!("{fn_name}({args})",)
                        }
                    }
                }
            }
            TypeTree::Variant { variant, generics, params } => {
                let generics = non_default_generics(db, (*variant).into(), generics);
                let generics_str = match generics.is_empty() {
                    true => String::new(),
                    false => {
                        let generics =
                            generics.iter().map(|it| type_path(sema_scope, it)).join(", ");
                        format!("::<{generics}>")
                    }
                };
                let inner = match variant.kind(db) {
                    StructKind::Tuple => {
                        let args = params.iter().map(|f| f.gen_source_code(sema_scope)).join(", ");
                        format!("{generics_str}({args})")
                    }
                    StructKind::Record => {
                        let fields = variant.fields(db);
                        let args = params
                            .iter()
                            .zip(fields.iter())
                            .map(|(a, f)| {
                                format!(
                                    "{}: {}",
                                    f.name(db).display(db.upcast()).to_string(),
                                    a.gen_source_code(sema_scope)
                                )
                            })
                            .join(", ");
                        format!("{generics_str}{{ {args} }}")
                    }
                    StructKind::Unit => generics_str,
                };

                let prefix = mod_item_path_str(sema_scope, &ModuleDef::Variant(*variant));
                format!("{prefix}{inner}")
            }
            TypeTree::Struct { strukt, generics, params } => {
                let generics = non_default_generics(db, (*strukt).into(), generics);
                let inner = match strukt.kind(db) {
                    StructKind::Tuple => {
                        let args = params.iter().map(|a| a.gen_source_code(sema_scope)).join(", ");
                        format!("({args})")
                    }
                    StructKind::Record => {
                        let fields = strukt.fields(db);
                        let args = params
                            .iter()
                            .zip(fields.iter())
                            .map(|(a, f)| {
                                format!(
                                    "{}: {}",
                                    f.name(db).display(db.upcast()).to_string(),
                                    a.gen_source_code(sema_scope)
                                )
                            })
                            .join(", ");
                        format!(" {{ {args} }}")
                    }
                    StructKind::Unit => match generics.is_empty() {
                        true => String::new(),
                        false => {
                            let generics =
                                generics.iter().map(|it| type_path(sema_scope, it)).join(", ");
                            format!("::<{generics}>")
                        }
                    },
                };

                let prefix = mod_item_path_str(sema_scope, &ModuleDef::Adt(Adt::Struct(*strukt)));
                format!("{prefix}{inner}")
            }
            TypeTree::Field { type_tree, field } => {
                let strukt = type_tree.gen_source_code(sema_scope);
                let field = field.name(db).display(db.upcast()).to_string();
                format!("{strukt}.{field}")
            }
            TypeTree::Reference(type_tree) => {
                let inner = type_tree.gen_source_code(sema_scope);
                format!("&{inner}")
            }
        }
    }

    /// Get type of the type tree.
    ///
    /// Same as getting the type of root node
    pub fn ty(&self, db: &dyn HirDatabase) -> Type {
        match self {
            TypeTree::Const(it) => it.ty(db),
            TypeTree::Static(it) => it.ty(db),
            TypeTree::Local(it) => it.ty(db),
            TypeTree::ConstParam(it) => it.ty(db),
            TypeTree::FamousType { ty, .. } => ty.clone(),
            TypeTree::Function { func, generics, params } => match func.has_self_param(db) {
                true => func.ret_type_with_generics(
                    db,
                    params[0].ty(db).type_arguments().chain(generics.iter().cloned()),
                ),
                false => func.ret_type_with_generics(db, generics.iter().cloned()),
            },
            TypeTree::Variant { variant, generics, .. } => {
                variant.parent_enum(db).ty_with_generics(db, generics.iter().cloned())
            }
            TypeTree::Struct { strukt, generics, .. } => {
                strukt.ty_with_generics(db, generics.iter().cloned())
            }
            TypeTree::Field { type_tree, field } => {
                field.ty_with_generics(db, type_tree.ty(db).type_arguments())
            }
            TypeTree::Reference(it) => it.ty(db),
        }
    }

    /// List the traits used in type tree
    pub fn traits_used(&self, db: &dyn HirDatabase) -> Vec<Trait> {
        let mut res = Vec::new();

        match self {
            TypeTree::Function { func, params, .. } => {
                res.extend(params.iter().flat_map(|it| it.traits_used(db)));
                if let Some(it) = func.as_assoc_item(db) {
                    if let Some(it) = it.containing_trait_or_trait_impl(db) {
                        res.push(it);
                    }
                }
            }
            _ => (),
        }

        res
    }
}

//! Type tree for term search

use hir_def::find_path::PrefixKind;
use hir_expand::mod_path::ModPath;
use hir_ty::{
    db::HirDatabase,
    display::{DisplaySourceCodeError, HirDisplay},
};
use itertools::Itertools;

use crate::{
    Adt, AsAssocItem, Const, ConstParam, Field, Function, GenericDef, Local, ModuleDef,
    SemanticsScope, Static, Struct, StructKind, Trait, Type, Variant,
};

/// Helper function to get path to `ModuleDef`
fn mod_item_path(
    sema_scope: &SemanticsScope<'_>,
    def: &ModuleDef,
    prefer_no_std: bool,
    prefer_prelude: bool,
) -> Option<ModPath> {
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
        Some(0..=1) | None => m.find_use_path(db.upcast(), *def, prefer_no_std, prefer_prelude),
        Some(_) => m.find_use_path_prefixed(
            db.upcast(),
            *def,
            PrefixKind::ByCrate,
            prefer_no_std,
            prefer_prelude,
        ),
    }
}

/// Helper function to get path to `ModuleDef` as string
fn mod_item_path_str(
    sema_scope: &SemanticsScope<'_>,
    def: &ModuleDef,
    prefer_no_std: bool,
    prefer_prelude: bool,
) -> Result<String, DisplaySourceCodeError> {
    let path = mod_item_path(sema_scope, def, prefer_no_std, prefer_prelude);
    path.map(|it| it.display(sema_scope.db.upcast()).to_string())
        .ok_or(DisplaySourceCodeError::PathNotFound)
}

/// Helper function to get path to `Type`
fn type_path(
    sema_scope: &SemanticsScope<'_>,
    ty: &Type,
    prefer_no_std: bool,
    prefer_prelude: bool,
) -> Result<String, DisplaySourceCodeError> {
    let db = sema_scope.db;
    let m = sema_scope.module();

    match ty.as_adt() {
        Some(adt) => {
            let ty_name = ty.display_source_code(db, m.id, true)?;

            let mut path =
                mod_item_path(sema_scope, &ModuleDef::Adt(adt), prefer_no_std, prefer_prelude)
                    .unwrap();
            path.pop_segment();
            let path = path.display(db.upcast()).to_string();
            let res = match path.is_empty() {
                true => ty_name,
                false => format!("{path}::{ty_name}"),
            };
            Ok(res)
        }
        None => ty.display_source_code(db, m.id, true),
    }
}

/// Helper function to filter out generic parameters that are default
fn non_default_generics(db: &dyn HirDatabase, def: GenericDef, generics: &[Type]) -> Vec<Type> {
    def.type_or_const_params(db)
        .into_iter()
        .filter_map(|it| it.as_type_param(db))
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
pub enum Expr {
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
    /// Function call (does not take self param)
    Function { func: Function, generics: Vec<Type>, params: Vec<Expr> },
    /// Method call (has self param)
    Method { func: Function, generics: Vec<Type>, target: Box<Expr>, params: Vec<Expr> },
    /// Enum variant construction
    Variant { variant: Variant, generics: Vec<Type>, params: Vec<Expr> },
    /// Struct construction
    Struct { strukt: Struct, generics: Vec<Type>, params: Vec<Expr> },
    /// Tuple construction
    Tuple { ty: Type, params: Vec<Expr> },
    /// Struct field access
    Field { expr: Box<Expr>, field: Field },
    /// Passing type as reference (with `&`)
    Reference(Box<Expr>),
    /// Indicates possibility of many different options that all evaluate to `ty`
    Many(Type),
}

impl Expr {
    /// Generate source code for type tree.
    ///
    /// Note that trait imports are not added to generated code.
    /// To make sure that the code is valid, callee has to also ensure that all the traits listed
    /// by `traits_used` method are also imported.
    pub fn gen_source_code(
        &self,
        sema_scope: &SemanticsScope<'_>,
        many_formatter: &mut dyn FnMut(&Type) -> String,
        prefer_no_std: bool,
        prefer_prelude: bool,
    ) -> Result<String, DisplaySourceCodeError> {
        let db = sema_scope.db;
        let mod_item_path_str = |s, def| mod_item_path_str(s, def, prefer_no_std, prefer_prelude);
        match self {
            Expr::Const(it) => mod_item_path_str(sema_scope, &ModuleDef::Const(*it)),
            Expr::Static(it) => mod_item_path_str(sema_scope, &ModuleDef::Static(*it)),
            Expr::Local(it) => Ok(it.name(db).display(db.upcast()).to_string()),
            Expr::ConstParam(it) => Ok(it.name(db).display(db.upcast()).to_string()),
            Expr::FamousType { value, .. } => Ok(value.to_string()),
            Expr::Function { func, params, .. } => {
                let args = params
                    .iter()
                    .map(|f| {
                        f.gen_source_code(sema_scope, many_formatter, prefer_no_std, prefer_prelude)
                    })
                    .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                    .into_iter()
                    .join(", ");

                match func.as_assoc_item(db).map(|it| it.container(db)) {
                    Some(container) => {
                        let container_name = match container {
                            crate::AssocItemContainer::Trait(trait_) => {
                                mod_item_path_str(sema_scope, &ModuleDef::Trait(trait_))?
                            }
                            crate::AssocItemContainer::Impl(imp) => {
                                let self_ty = imp.self_ty(db);
                                // Should it be guaranteed that `mod_item_path` always exists?
                                match self_ty.as_adt().and_then(|adt| {
                                    mod_item_path(
                                        sema_scope,
                                        &adt.into(),
                                        prefer_no_std,
                                        prefer_prelude,
                                    )
                                }) {
                                    Some(path) => path.display(sema_scope.db.upcast()).to_string(),
                                    None => self_ty.display(db).to_string(),
                                }
                            }
                        };
                        let fn_name = func.name(db).display(db.upcast()).to_string();
                        Ok(format!("{container_name}::{fn_name}({args})"))
                    }
                    None => {
                        let fn_name = mod_item_path_str(sema_scope, &ModuleDef::Function(*func))?;
                        Ok(format!("{fn_name}({args})"))
                    }
                }
            }
            Expr::Method { func, target, params, .. } => {
                if target.contains_many_in_illegal_pos() {
                    return Ok(many_formatter(&target.ty(db)));
                }

                let func_name = func.name(db).display(db.upcast()).to_string();
                let self_param = func.self_param(db).unwrap();
                let target = target.gen_source_code(
                    sema_scope,
                    many_formatter,
                    prefer_no_std,
                    prefer_prelude,
                )?;
                let args = params
                    .iter()
                    .map(|f| {
                        f.gen_source_code(sema_scope, many_formatter, prefer_no_std, prefer_prelude)
                    })
                    .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                    .into_iter()
                    .join(", ");

                match func.as_assoc_item(db).and_then(|it| it.container_or_implemented_trait(db)) {
                    Some(trait_) => {
                        let trait_name = mod_item_path_str(sema_scope, &ModuleDef::Trait(trait_))?;
                        let target = match self_param.access(db) {
                            crate::Access::Shared => format!("&{target}"),
                            crate::Access::Exclusive => format!("&mut {target}"),
                            crate::Access::Owned => target,
                        };
                        let res = match args.is_empty() {
                            true => format!("{trait_name}::{func_name}({target})",),
                            false => format!("{trait_name}::{func_name}({target}, {args})",),
                        };
                        Ok(res)
                    }
                    None => Ok(format!("{target}.{func_name}({args})")),
                }
            }
            Expr::Variant { variant, generics, params } => {
                let generics = non_default_generics(db, (*variant).into(), generics);
                let generics_str = match generics.is_empty() {
                    true => String::new(),
                    false => {
                        let generics = generics
                            .iter()
                            .map(|it| type_path(sema_scope, it, prefer_no_std, prefer_prelude))
                            .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                            .into_iter()
                            .join(", ");
                        format!("::<{generics}>")
                    }
                };
                let inner = match variant.kind(db) {
                    StructKind::Tuple => {
                        let args = params
                            .iter()
                            .map(|f| {
                                f.gen_source_code(
                                    sema_scope,
                                    many_formatter,
                                    prefer_no_std,
                                    prefer_prelude,
                                )
                            })
                            .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                            .into_iter()
                            .join(", ");
                        format!("{generics_str}({args})")
                    }
                    StructKind::Record => {
                        let fields = variant.fields(db);
                        let args = params
                            .iter()
                            .zip(fields.iter())
                            .map(|(a, f)| {
                                let tmp = format!(
                                    "{}: {}",
                                    f.name(db).display(db.upcast()),
                                    a.gen_source_code(
                                        sema_scope,
                                        many_formatter,
                                        prefer_no_std,
                                        prefer_prelude
                                    )?
                                );
                                Ok(tmp)
                            })
                            .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                            .into_iter()
                            .join(", ");
                        format!("{generics_str}{{ {args} }}")
                    }
                    StructKind::Unit => generics_str,
                };

                let prefix = mod_item_path_str(sema_scope, &ModuleDef::Variant(*variant))?;
                Ok(format!("{prefix}{inner}"))
            }
            Expr::Struct { strukt, generics, params } => {
                let generics = non_default_generics(db, (*strukt).into(), generics);
                let inner = match strukt.kind(db) {
                    StructKind::Tuple => {
                        let args = params
                            .iter()
                            .map(|a| {
                                a.gen_source_code(
                                    sema_scope,
                                    many_formatter,
                                    prefer_no_std,
                                    prefer_prelude,
                                )
                            })
                            .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                            .into_iter()
                            .join(", ");
                        format!("({args})")
                    }
                    StructKind::Record => {
                        let fields = strukt.fields(db);
                        let args = params
                            .iter()
                            .zip(fields.iter())
                            .map(|(a, f)| {
                                let tmp = format!(
                                    "{}: {}",
                                    f.name(db).display(db.upcast()),
                                    a.gen_source_code(
                                        sema_scope,
                                        many_formatter,
                                        prefer_no_std,
                                        prefer_prelude
                                    )?
                                );
                                Ok(tmp)
                            })
                            .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                            .into_iter()
                            .join(", ");
                        format!(" {{ {args} }}")
                    }
                    StructKind::Unit => match generics.is_empty() {
                        true => String::new(),
                        false => {
                            let generics = generics
                                .iter()
                                .map(|it| type_path(sema_scope, it, prefer_no_std, prefer_prelude))
                                .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                                .into_iter()
                                .join(", ");
                            format!("::<{generics}>")
                        }
                    },
                };

                let prefix = mod_item_path_str(sema_scope, &ModuleDef::Adt(Adt::Struct(*strukt)))?;
                Ok(format!("{prefix}{inner}"))
            }
            Expr::Tuple { params, .. } => {
                let args = params
                    .iter()
                    .map(|a| {
                        a.gen_source_code(sema_scope, many_formatter, prefer_no_std, prefer_prelude)
                    })
                    .collect::<Result<Vec<String>, DisplaySourceCodeError>>()?
                    .into_iter()
                    .join(", ");
                let res = format!("({args})");
                Ok(res)
            }
            Expr::Field { expr, field } => {
                if expr.contains_many_in_illegal_pos() {
                    return Ok(many_formatter(&expr.ty(db)));
                }

                let strukt = expr.gen_source_code(
                    sema_scope,
                    many_formatter,
                    prefer_no_std,
                    prefer_prelude,
                )?;
                let field = field.name(db).display(db.upcast()).to_string();
                Ok(format!("{strukt}.{field}"))
            }
            Expr::Reference(expr) => {
                if expr.contains_many_in_illegal_pos() {
                    return Ok(many_formatter(&expr.ty(db)));
                }

                let inner = expr.gen_source_code(
                    sema_scope,
                    many_formatter,
                    prefer_no_std,
                    prefer_prelude,
                )?;
                Ok(format!("&{inner}"))
            }
            Expr::Many(ty) => Ok(many_formatter(ty)),
        }
    }

    /// Get type of the type tree.
    ///
    /// Same as getting the type of root node
    pub fn ty(&self, db: &dyn HirDatabase) -> Type {
        match self {
            Expr::Const(it) => it.ty(db),
            Expr::Static(it) => it.ty(db),
            Expr::Local(it) => it.ty(db),
            Expr::ConstParam(it) => it.ty(db),
            Expr::FamousType { ty, .. } => ty.clone(),
            Expr::Function { func, generics, .. } => {
                func.ret_type_with_args(db, generics.iter().cloned())
            }
            Expr::Method { func, generics, target, .. } => func.ret_type_with_args(
                db,
                target.ty(db).type_arguments().chain(generics.iter().cloned()),
            ),
            Expr::Variant { variant, generics, .. } => {
                Adt::from(variant.parent_enum(db)).ty_with_args(db, generics.iter().cloned())
            }
            Expr::Struct { strukt, generics, .. } => {
                Adt::from(*strukt).ty_with_args(db, generics.iter().cloned())
            }
            Expr::Tuple { ty, .. } => ty.clone(),
            Expr::Field { expr, field } => field.ty_with_args(db, expr.ty(db).type_arguments()),
            Expr::Reference(it) => it.ty(db),
            Expr::Many(ty) => ty.clone(),
        }
    }

    /// List the traits used in type tree
    pub fn traits_used(&self, db: &dyn HirDatabase) -> Vec<Trait> {
        let mut res = Vec::new();

        if let Expr::Method { func, params, .. } = self {
            res.extend(params.iter().flat_map(|it| it.traits_used(db)));
            if let Some(it) = func.as_assoc_item(db) {
                if let Some(it) = it.container_or_implemented_trait(db) {
                    res.push(it);
                }
            }
        }

        res
    }

    /// Check in the tree contains `Expr::Many` variant in illegal place to insert `todo`,
    /// `unimplemented` or similar macro
    ///
    /// Some examples are following
    /// ```no_compile
    /// macro!().foo
    /// macro!().bar()
    /// &macro!()
    /// ```
    fn contains_many_in_illegal_pos(&self) -> bool {
        match self {
            Expr::Method { target, .. } => target.contains_many_in_illegal_pos(),
            Expr::Field { expr, .. } => expr.contains_many_in_illegal_pos(),
            Expr::Reference(target) => target.is_many(),
            Expr::Many(_) => true,
            _ => false,
        }
    }

    /// Helper function to check if outermost type tree is `Expr::Many` variant
    pub fn is_many(&self) -> bool {
        matches!(self, Expr::Many(_))
    }
}

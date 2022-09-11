// Some ideas for future improvements:
// - Support replacing aliases which are used in expressions, e.g. `A::new()`.
// - Remove unused aliases if there are no longer any users, see inline_call.rs.

use hir::{HasSource, PathResolution};
use ide_db::{defs::Definition, search::FileReference};
use itertools::Itertools;
use std::collections::HashMap;
use syntax::{
    ast::{self, make, HasGenericParams, HasName},
    ted, AstNode, NodeOrToken, SyntaxNode,
};

use crate::{
    assist_context::{AssistContext, Assists},
    AssistId, AssistKind,
};

// Assist: inline_type_alias_uses
//
// Inline a type alias into all of its uses where possible.
//
// ```
// type $0A = i32;
// fn id(x: A) -> A {
//     x
// };
// fn foo() {
//     let _: A = 3;
// }
// ```
// ->
// ```
// type A = i32;
// fn id(x: i32) -> i32 {
//     x
// };
// fn foo() {
//     let _: i32 = 3;
// }
pub(crate) fn inline_type_alias_uses(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let name = ctx.find_node_at_offset::<ast::Name>()?;
    let ast_alias = name.syntax().parent().and_then(ast::TypeAlias::cast)?;

    let hir_alias = ctx.sema.to_def(&ast_alias)?;
    let concrete_type = ast_alias.ty()?;

    let usages = Definition::TypeAlias(hir_alias).usages(&ctx.sema);
    if !usages.at_least_one() {
        return None;
    }

    // until this is ok

    acc.add(
        AssistId("inline_type_alias_uses", AssistKind::RefactorInline),
        "Inline type alias into all uses",
        name.syntax().text_range(),
        |builder| {
            let usages = usages.all();

            let mut inline_refs_for_file = |file_id, refs: Vec<FileReference>| {
                builder.edit_file(file_id);

                let path_types: Vec<ast::PathType> = refs
                    .into_iter()
                    .filter_map(|file_ref| match file_ref.name {
                        ast::NameLike::NameRef(path_type) => {
                            path_type.syntax().ancestors().nth(3).and_then(ast::PathType::cast)
                        }
                        _ => None,
                    })
                    .collect();

                for (target, replacement) in path_types.into_iter().filter_map(|path_type| {
                    let replacement = inline(&ast_alias, &path_type)?.to_text(&concrete_type);
                    let target = path_type.syntax().text_range();
                    Some((target, replacement))
                }) {
                    builder.replace(target, replacement);
                }
            };

            for (file_id, refs) in usages.into_iter() {
                inline_refs_for_file(file_id, refs);
            }
        },
    )
}

// Assist: inline_type_alias
//
// Replace a type alias with its concrete type.
//
// ```
// type A<T = u32> = Vec<T>;
//
// fn main() {
//     let a: $0A;
// }
// ```
// ->
// ```
// type A<T = u32> = Vec<T>;
//
// fn main() {
//     let a: Vec<u32>;
// }
// ```
pub(crate) fn inline_type_alias(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let alias_instance = ctx.find_node_at_offset::<ast::PathType>()?;
    let concrete_type;
    let replacement;
    match alias_instance.path()?.as_single_name_ref() {
        Some(nameref) if nameref.Self_token().is_some() => {
            match ctx.sema.resolve_path(&alias_instance.path()?)? {
                PathResolution::SelfType(imp) => {
                    concrete_type = imp.source(ctx.db())?.value.self_ty()?;
                }
                // FIXME: should also work in ADT definitions
                _ => return None,
            }

            replacement = Replacement::Plain;
        }
        _ => {
            let alias = get_type_alias(&ctx, &alias_instance)?;
            concrete_type = alias.ty()?;
            replacement = inline(&alias, &alias_instance)?;
        }
    }

    let target = alias_instance.syntax().text_range();

    acc.add(
        AssistId("inline_type_alias", AssistKind::RefactorInline),
        "Inline type alias",
        target,
        |builder| builder.replace(target, replacement.to_text(&concrete_type)),
    )
}

impl Replacement {
    fn to_text(&self, concrete_type: &ast::Type) -> String {
        match self {
            Replacement::Generic { lifetime_map, const_and_type_map } => {
                create_replacement(&lifetime_map, &const_and_type_map, &concrete_type)
            }
            Replacement::Plain => concrete_type.to_string(),
        }
    }
}

enum Replacement {
    Generic { lifetime_map: LifetimeMap, const_and_type_map: ConstAndTypeMap },
    Plain,
}

fn inline(alias_def: &ast::TypeAlias, alias_instance: &ast::PathType) -> Option<Replacement> {
    let repl = if let Some(alias_generics) = alias_def.generic_param_list() {
        if alias_generics.generic_params().next().is_none() {
            cov_mark::hit!(no_generics_params);
            return None;
        }
        let instance_args =
            alias_instance.syntax().descendants().find_map(ast::GenericArgList::cast);

        Replacement::Generic {
            lifetime_map: LifetimeMap::new(&instance_args, &alias_generics)?,
            const_and_type_map: ConstAndTypeMap::new(&instance_args, &alias_generics)?,
        }
    } else {
        Replacement::Plain
    };
    Some(repl)
}

struct LifetimeMap(HashMap<String, ast::Lifetime>);

impl LifetimeMap {
    fn new(
        instance_args: &Option<ast::GenericArgList>,
        alias_generics: &ast::GenericParamList,
    ) -> Option<Self> {
        let mut inner = HashMap::new();

        let wildcard_lifetime = make::lifetime("'_");
        let lifetimes = alias_generics
            .lifetime_params()
            .filter_map(|lp| lp.lifetime())
            .map(|l| l.to_string())
            .collect_vec();

        for lifetime in &lifetimes {
            inner.insert(lifetime.to_string(), wildcard_lifetime.clone());
        }

        if let Some(instance_generic_args_list) = &instance_args {
            for (index, lifetime) in instance_generic_args_list
                .lifetime_args()
                .filter_map(|arg| arg.lifetime())
                .enumerate()
            {
                let key = match lifetimes.get(index) {
                    Some(key) => key,
                    None => {
                        cov_mark::hit!(too_many_lifetimes);
                        return None;
                    }
                };

                inner.insert(key.clone(), lifetime);
            }
        }

        Some(Self(inner))
    }
}

struct ConstAndTypeMap(HashMap<String, SyntaxNode>);

impl ConstAndTypeMap {
    fn new(
        instance_args: &Option<ast::GenericArgList>,
        alias_generics: &ast::GenericParamList,
    ) -> Option<Self> {
        let mut inner = HashMap::new();
        let instance_generics = generic_args_to_const_and_type_generics(instance_args);
        let alias_generics = generic_param_list_to_const_and_type_generics(&alias_generics);

        if instance_generics.len() > alias_generics.len() {
            cov_mark::hit!(too_many_generic_args);
            return None;
        }

        // Any declaration generics that don't have a default value must have one
        // provided by the instance.
        for (i, declaration_generic) in alias_generics.iter().enumerate() {
            let key = declaration_generic.replacement_key()?;

            if let Some(instance_generic) = instance_generics.get(i) {
                inner.insert(key, instance_generic.replacement_value()?);
            } else if let Some(value) = declaration_generic.replacement_value() {
                inner.insert(key, value);
            } else {
                cov_mark::hit!(missing_replacement_param);
                return None;
            }
        }

        Some(Self(inner))
    }
}

/// This doesn't attempt to ensure specified generics are compatible with those
/// required by the type alias, other than lifetimes which must either all be
/// specified or all omitted. It will replace TypeArgs with ConstArgs and vice
/// versa if they're in the wrong position. It supports partially specified
/// generics.
///
/// 1. Map the provided instance's generic args to the type alias's generic
///    params:
///
///    ```
///    type A<'a, const N: usize, T = u64> = &'a [T; N];
///          ^ alias generic params
///    let a: A<100>;
///            ^ instance generic args
///    ```
///
///    generic['a] = '_ due to omission
///    generic[N] = 100 due to the instance arg
///    generic[T] = u64 due to the default param
///
/// 2. Copy the concrete type and substitute in each found mapping:
///
///    &'_ [u64; 100]
///
/// 3. Remove wildcard lifetimes entirely:
///
///    &[u64; 100]
fn create_replacement(
    lifetime_map: &LifetimeMap,
    const_and_type_map: &ConstAndTypeMap,
    concrete_type: &ast::Type,
) -> String {
    let updated_concrete_type = concrete_type.clone_for_update();
    let mut replacements = Vec::new();
    let mut removals = Vec::new();

    for syntax in updated_concrete_type.syntax().descendants() {
        let syntax_string = syntax.to_string();
        let syntax_str = syntax_string.as_str();

        if let Some(old_lifetime) = ast::Lifetime::cast(syntax.clone()) {
            if let Some(new_lifetime) = lifetime_map.0.get(&old_lifetime.to_string()) {
                if new_lifetime.text() == "'_" {
                    removals.push(NodeOrToken::Node(syntax.clone()));

                    if let Some(ws) = syntax.next_sibling_or_token() {
                        removals.push(ws.clone());
                    }

                    continue;
                }

                replacements.push((syntax.clone(), new_lifetime.syntax().clone_for_update()));
            }
        } else if let Some(replacement_syntax) = const_and_type_map.0.get(syntax_str) {
            let new_string = replacement_syntax.to_string();
            let new = if new_string == "_" {
                make::wildcard_pat().syntax().clone_for_update()
            } else {
                replacement_syntax.clone_for_update()
            };

            replacements.push((syntax.clone(), new));
        }
    }

    for (old, new) in replacements {
        ted::replace(old, new);
    }

    for syntax in removals {
        ted::remove(syntax);
    }

    updated_concrete_type.to_string()
}

fn get_type_alias(ctx: &AssistContext<'_>, path: &ast::PathType) -> Option<ast::TypeAlias> {
    let resolved_path = ctx.sema.resolve_path(&path.path()?)?;

    // We need the generics in the correct order to be able to map any provided
    // instance generics to declaration generics. The `hir::TypeAlias` doesn't
    // keep the order, so we must get the `ast::TypeAlias` from the hir
    // definition.
    if let PathResolution::Def(hir::ModuleDef::TypeAlias(ta)) = resolved_path {
        Some(ctx.sema.source(ta)?.value)
    } else {
        None
    }
}

enum ConstOrTypeGeneric {
    ConstArg(ast::ConstArg),
    TypeArg(ast::TypeArg),
    ConstParam(ast::ConstParam),
    TypeParam(ast::TypeParam),
}

impl ConstOrTypeGeneric {
    fn replacement_key(&self) -> Option<String> {
        // Only params are used as replacement keys.
        match self {
            ConstOrTypeGeneric::ConstParam(cp) => Some(cp.name()?.to_string()),
            ConstOrTypeGeneric::TypeParam(tp) => Some(tp.name()?.to_string()),
            _ => None,
        }
    }

    fn replacement_value(&self) -> Option<SyntaxNode> {
        Some(match self {
            ConstOrTypeGeneric::ConstArg(ca) => ca.expr()?.syntax().clone(),
            ConstOrTypeGeneric::TypeArg(ta) => ta.syntax().clone(),
            ConstOrTypeGeneric::ConstParam(cp) => cp.default_val()?.syntax().clone(),
            ConstOrTypeGeneric::TypeParam(tp) => tp.default_type()?.syntax().clone(),
        })
    }
}

fn generic_param_list_to_const_and_type_generics(
    generics: &ast::GenericParamList,
) -> Vec<ConstOrTypeGeneric> {
    let mut others = Vec::new();

    for param in generics.generic_params() {
        match param {
            ast::GenericParam::LifetimeParam(_) => {}
            ast::GenericParam::ConstParam(cp) => {
                others.push(ConstOrTypeGeneric::ConstParam(cp));
            }
            ast::GenericParam::TypeParam(tp) => others.push(ConstOrTypeGeneric::TypeParam(tp)),
        }
    }

    others
}

fn generic_args_to_const_and_type_generics(
    generics: &Option<ast::GenericArgList>,
) -> Vec<ConstOrTypeGeneric> {
    let mut others = Vec::new();

    // It's fine for there to be no instance generics because the declaration
    // might have default values or they might be inferred.
    if let Some(generics) = generics {
        for arg in generics.generic_args() {
            match arg {
                ast::GenericArg::TypeArg(ta) => {
                    others.push(ConstOrTypeGeneric::TypeArg(ta));
                }
                ast::GenericArg::ConstArg(ca) => {
                    others.push(ConstOrTypeGeneric::ConstArg(ca));
                }
                _ => {}
            }
        }
    }

    others
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn empty_generic_params() {
        cov_mark::check!(no_generics_params);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<> = T;
fn main() {
    let a: $0A<u32>;
}
            "#,
        );
    }

    #[test]
    fn too_many_generic_args() {
        cov_mark::check!(too_many_generic_args);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<T> = T;
fn main() {
    let a: $0A<u32, u64>;
}
            "#,
        );
    }

    #[test]
    fn too_many_lifetimes() {
        cov_mark::check!(too_many_lifetimes);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<'a> = &'a &'b u32;
fn f<'a>() {
    let a: $0A<'a, 'b> = 0;
}
"#,
        );
    }

    // This must be supported in order to support "inline_alias_to_users" or
    // whatever it will be called.
    #[test]
    fn alias_as_expression_ignored() {
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A = Vec<u32>;
fn main() {
    let a: A = $0A::new();
}
"#,
        );
    }

    #[test]
    fn primitive_arg() {
        check_assist(
            inline_type_alias,
            r#"
type A<T> = T;
fn main() {
    let a: $0A<u32> = 0;
}
"#,
            r#"
type A<T> = T;
fn main() {
    let a: u32 = 0;
}
"#,
        );
    }

    #[test]
    fn no_generic_replacements() {
        check_assist(
            inline_type_alias,
            r#"
type A = Vec<u32>;
fn main() {
    let a: $0A;
}
"#,
            r#"
type A = Vec<u32>;
fn main() {
    let a: Vec<u32>;
}
"#,
        );
    }

    #[test]
    fn param_expression() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize = { 1 }> = [u32; N];
fn main() {
    let a: $0A;
}
"#,
            r#"
type A<const N: usize = { 1 }> = [u32; N];
fn main() {
    let a: [u32; { 1 }];
}
"#,
        );
    }

    #[test]
    fn param_default_value() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize = 1> = [u32; N];
fn main() {
    let a: $0A;
}
"#,
            r#"
type A<const N: usize = 1> = [u32; N];
fn main() {
    let a: [u32; 1];
}
"#,
        );
    }

    #[test]
    fn all_param_types() {
        check_assist(
            inline_type_alias,
            r#"
struct Struct<const C: usize>;
type A<'inner1, 'outer1, Outer1, const INNER1: usize, Inner1: Clone, const OUTER1: usize> = (Struct<INNER1>, Struct<OUTER1>, Outer1, &'inner1 (), Inner1, &'outer1 ());
fn foo<'inner2, 'outer2, Outer2, const INNER2: usize, Inner2, const OUTER2: usize>() {
    let a: $0A<'inner2, 'outer2, Outer2, INNER2, Inner2, OUTER2>;
}
"#,
            r#"
struct Struct<const C: usize>;
type A<'inner1, 'outer1, Outer1, const INNER1: usize, Inner1: Clone, const OUTER1: usize> = (Struct<INNER1>, Struct<OUTER1>, Outer1, &'inner1 (), Inner1, &'outer1 ());
fn foo<'inner2, 'outer2, Outer2, const INNER2: usize, Inner2, const OUTER2: usize>() {
    let a: (Struct<INNER2>, Struct<OUTER2>, Outer2, &'inner2 (), Inner2, &'outer2 ());
}
"#,
        );
    }

    #[test]
    fn omitted_lifetimes() {
        check_assist(
            inline_type_alias,
            r#"
type A<'l, 'r> = &'l &'r u32;
fn main() {
    let a: $0A;
}
"#,
            r#"
type A<'l, 'r> = &'l &'r u32;
fn main() {
    let a: &&u32;
}
"#,
        );
    }

    #[test]
    fn omitted_type() {
        check_assist(
            inline_type_alias,
            r#"
type A<'r, 'l, T = u32> = &'l std::collections::HashMap<&'r str, T>;
fn main() {
    let a: $0A<'_, '_>;
}
"#,
            r#"
type A<'r, 'l, T = u32> = &'l std::collections::HashMap<&'r str, T>;
fn main() {
    let a: &std::collections::HashMap<&str, u32>;
}
"#,
        );
    }

    #[test]
    fn omitted_everything() {
        check_assist(
            inline_type_alias,
            r#"
type A<'r, 'l, T = u32> = &'l std::collections::HashMap<&'r str, T>;
fn main() {
    let v = std::collections::HashMap<&str, u32>;
    let a: $0A = &v;
}
"#,
            r#"
type A<'r, 'l, T = u32> = &'l std::collections::HashMap<&'r str, T>;
fn main() {
    let v = std::collections::HashMap<&str, u32>;
    let a: &std::collections::HashMap<&str, u32> = &v;
}
"#,
        );
    }

    // This doesn't actually cause the GenericArgsList to contain a AssocTypeArg.
    #[test]
    fn arg_associated_type() {
        check_assist(
            inline_type_alias,
            r#"
trait Tra { type Assoc; fn a(); }
struct Str {}
impl Tra for Str {
    type Assoc = u32;
    fn a() {
        type A<T> = Vec<T>;
        let a: $0A<Self::Assoc>;
    }
}
"#,
            r#"
trait Tra { type Assoc; fn a(); }
struct Str {}
impl Tra for Str {
    type Assoc = u32;
    fn a() {
        type A<T> = Vec<T>;
        let a: Vec<Self::Assoc>;
    }
}
"#,
        );
    }

    #[test]
    fn param_default_associated_type() {
        check_assist(
            inline_type_alias,
            r#"
trait Tra { type Assoc; fn a() }
struct Str {}
impl Tra for Str {
    type Assoc = u32;
    fn a() {
        type A<T = Self::Assoc> = Vec<T>;
        let a: $0A;
    }
}
"#,
            r#"
trait Tra { type Assoc; fn a() }
struct Str {}
impl Tra for Str {
    type Assoc = u32;
    fn a() {
        type A<T = Self::Assoc> = Vec<T>;
        let a: Vec<Self::Assoc>;
    }
}
"#,
        );
    }

    #[test]
    fn function_pointer() {
        check_assist(
            inline_type_alias,
            r#"
type A = fn(u32);
fn foo(a: u32) {}
fn main() {
    let a: $0A = foo;
}
"#,
            r#"
type A = fn(u32);
fn foo(a: u32) {}
fn main() {
    let a: fn(u32) = foo;
}
"#,
        );
    }

    #[test]
    fn closure() {
        check_assist(
            inline_type_alias,
            r#"
type A = Box<dyn FnOnce(u32) -> u32>;
fn main() {
    let a: $0A = Box::new(|_| 0);
}
"#,
            r#"
type A = Box<dyn FnOnce(u32) -> u32>;
fn main() {
    let a: Box<dyn FnOnce(u32) -> u32> = Box::new(|_| 0);
}
"#,
        );
    }

    // Type aliases can't be used in traits, but someone might use the assist to
    // fix the error.
    #[test]
    fn bounds() {
        check_assist(
            inline_type_alias,
            r#"type A = std::io::Write; fn f<T>() where T: $0A {}"#,
            r#"type A = std::io::Write; fn f<T>() where T: std::io::Write {}"#,
        );
    }

    #[test]
    fn function_parameter() {
        check_assist(
            inline_type_alias,
            r#"
type A = std::io::Write;
fn f(a: impl $0A) {}
"#,
            r#"
type A = std::io::Write;
fn f(a: impl std::io::Write) {}
"#,
        );
    }

    #[test]
    fn arg_expression() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: $0A<{ 1 + 1 }>;
}
"#,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: [u32; { 1 + 1 }];
}
"#,
        )
    }

    #[test]
    fn alias_instance_generic_path() {
        check_assist(
            inline_type_alias,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: $0A<u32::MAX>;
}
"#,
            r#"
type A<const N: usize> = [u32; N];
fn main() {
    let a: [u32; u32::MAX];
}
"#,
        )
    }

    #[test]
    fn generic_type() {
        check_assist(
            inline_type_alias,
            r#"
type A = String;
fn f(a: Vec<$0A>) {}
"#,
            r#"
type A = String;
fn f(a: Vec<String>) {}
"#,
        );
    }

    #[test]
    fn missing_replacement_param() {
        cov_mark::check!(missing_replacement_param);
        check_assist_not_applicable(
            inline_type_alias,
            r#"
type A<U> = Vec<T>;
fn main() {
    let a: $0A;
}
"#,
        );
    }

    #[test]
    fn full_path_type_is_replaced() {
        check_assist(
            inline_type_alias,
            r#"
mod foo {
    pub type A = String;
}
fn main() {
    let a: foo::$0A;
}
"#,
            r#"
mod foo {
    pub type A = String;
}
fn main() {
    let a: String;
}
"#,
        );
    }

    #[test]
    fn inline_self_type() {
        check_assist(
            inline_type_alias,
            r#"
struct Strukt;

impl Strukt {
    fn new() -> Self$0 {}
}
"#,
            r#"
struct Strukt;

impl Strukt {
    fn new() -> Strukt {}
}
"#,
        );
        check_assist(
            inline_type_alias,
            r#"
struct Strukt<'a, T, const C: usize>(&'a [T; C]);

impl<T, const C: usize> Strukt<'_, T, C> {
    fn new() -> Self$0 {}
}
"#,
            r#"
struct Strukt<'a, T, const C: usize>(&'a [T; C]);

impl<T, const C: usize> Strukt<'_, T, C> {
    fn new() -> Strukt<'_, T, C> {}
}
"#,
        );
        check_assist(
            inline_type_alias,
            r#"
struct Strukt<'a, T, const C: usize>(&'a [T; C]);

trait Tr<'b, T> {}

impl<T, const C: usize> Tr<'static, u8> for Strukt<'_, T, C> {
    fn new() -> Self$0 {}
}
"#,
            r#"
struct Strukt<'a, T, const C: usize>(&'a [T; C]);

trait Tr<'b, T> {}

impl<T, const C: usize> Tr<'static, u8> for Strukt<'_, T, C> {
    fn new() -> Strukt<'_, T, C> {}
}
"#,
        );

        check_assist_not_applicable(
            inline_type_alias,
            r#"
trait Tr {
    fn new() -> Self$0;
}
"#,
        );
    }

    mod inline_type_alias_uses {
        use crate::{handlers::inline_type_alias::inline_type_alias_uses, tests::check_assist};

        #[test]
        fn inline_uses() {
            check_assist(
                inline_type_alias_uses,
                r#"
type $0A = u32;

fn foo() {
    let _: A = 3;
    let _: A = 4;
}
"#,
                r#"
type A = u32;

fn foo() {
    let _: u32 = 3;
    let _: u32 = 4;
}
"#,
            );
        }

        #[test]
        fn inline_uses_across_files() {
            check_assist(
                inline_type_alias_uses,
                r#"
//- /lib.rs
mod foo;
type $0T<E> = Vec<E>;
fn f() -> T<&str> {
    vec!["hello"]
}

//- /foo.rs
use super::T;
fn foo() {
    let _: T<i8> = Vec::new();
}
"#,
                r#"
//- /lib.rs
mod foo;
type T<E> = Vec<E>;
fn f() -> Vec<&str> {
    vec!["hello"]
}

//- /foo.rs
use super::T;
fn foo() {
    let _: Vec<i8> = Vec::new();
}
"#,
            );
        }

        #[test]
        fn inline_uses_across_files_2() {
            check_assist(
                inline_type_alias_uses,
                r#"
//- /lib.rs
mod foo;
type $0I = i32;

//- /foo.rs
use super::I;
fn foo() {
    let _: I = 0;
}
"#,
                r#"
use super::I;
fn foo() {
    let _: i32 = 0;
}
"#,
            );
        }
    }
}

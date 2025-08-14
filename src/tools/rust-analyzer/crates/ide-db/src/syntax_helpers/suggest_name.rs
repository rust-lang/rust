//! This module contains functions to suggest names for expressions, functions and other items

use std::{collections::hash_map::Entry, str::FromStr};

use hir::{Semantics, SemanticsScope};
use itertools::Itertools;
use rustc_hash::FxHashMap;
use stdx::to_lower_snake_case;
use syntax::{
    AstNode, Edition, SmolStr, SmolStrBuilder, ToSmolStr,
    ast::{self, HasName},
    match_ast,
};

use crate::RootDatabase;

/// Trait names, that will be ignored when in `impl Trait` and `dyn Trait`
const USELESS_TRAITS: &[&str] = &["Send", "Sync", "Copy", "Clone", "Eq", "PartialEq"];

/// Identifier names that won't be suggested, ever
///
/// **NOTE**: they all must be snake lower case
const USELESS_NAMES: &[&str] =
    &["new", "default", "option", "some", "none", "ok", "err", "str", "string", "from", "into"];

const USELESS_NAME_PREFIXES: &[&str] = &["from_", "with_", "into_"];

/// Generic types replaced by their first argument
///
/// # Examples
/// `Option<Name>` -> `Name`
/// `Result<User, Error>` -> `User`
const WRAPPER_TYPES: &[&str] = &["Box", "Arc", "Rc", "Option", "Result"];

/// Generic types replaced by a plural of their first argument.
///
/// # Examples
/// `Vec<Name>` -> "names"
const SEQUENCE_TYPES: &[&str] = &["Vec", "VecDeque", "LinkedList"];

/// Prefixes to strip from methods names
///
/// # Examples
/// `vec.as_slice()` -> `slice`
/// `args.into_config()` -> `config`
/// `bytes.to_vec()` -> `vec`
const USELESS_METHOD_PREFIXES: &[&str] = &["into_", "as_", "to_"];

/// Useless methods that are stripped from expression
///
/// # Examples
/// `var.name().to_string()` -> `var.name()`
const USELESS_METHODS: &[&str] = &[
    "to_string",
    "as_str",
    "to_owned",
    "as_ref",
    "clone",
    "cloned",
    "expect",
    "expect_none",
    "unwrap",
    "unwrap_none",
    "unwrap_or",
    "unwrap_or_default",
    "unwrap_or_else",
    "unwrap_unchecked",
    "iter",
    "into_iter",
    "iter_mut",
    "into_future",
];

/// Generator for new names
///
/// The generator keeps track of existing names and suggests new names that do
/// not conflict with existing names.
///
/// The generator will try to resolve conflicts by adding a numeric suffix to
/// the name, e.g. `a`, `a1`, `a2`, ...
///
/// # Examples
///
/// ```
/// # use ide_db::syntax_helpers::suggest_name::NameGenerator;
/// let mut generator = NameGenerator::default();
/// assert_eq!(generator.suggest_name("a"), "a");
/// assert_eq!(generator.suggest_name("a"), "a1");
///
/// assert_eq!(generator.suggest_name("b2"), "b2");
/// assert_eq!(generator.suggest_name("b"), "b3");
/// ```
#[derive(Debug, Default)]
pub struct NameGenerator {
    pool: FxHashMap<SmolStr, usize>,
}

impl NameGenerator {
    /// Create a new generator with existing names. When suggesting a name, it will
    /// avoid conflicts with existing names.
    pub fn new_with_names<'a>(existing_names: impl Iterator<Item = &'a str>) -> Self {
        let mut generator = Self::default();
        existing_names.for_each(|name| generator.insert(name));
        generator
    }

    pub fn new_from_scope_locals(scope: Option<SemanticsScope<'_>>) -> Self {
        let mut generator = Self::default();
        if let Some(scope) = scope {
            scope.process_all_names(&mut |name, scope| {
                if let hir::ScopeDef::Local(_) = scope {
                    generator.insert(name.as_str());
                }
            });
        }

        generator
    }

    /// Suggest a name without conflicts. If the name conflicts with existing names,
    /// it will try to resolve the conflict by adding a numeric suffix.
    pub fn suggest_name(&mut self, name: &str) -> SmolStr {
        let (prefix, suffix) = Self::split_numeric_suffix(name);
        let prefix = SmolStr::new(prefix);
        let suffix = suffix.unwrap_or(0);

        match self.pool.entry(prefix.clone()) {
            Entry::Vacant(entry) => {
                entry.insert(suffix);
                SmolStr::from_str(name).unwrap()
            }
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count = (*count + 1).max(suffix);

                let mut new_name = SmolStrBuilder::new();
                new_name.push_str(&prefix);
                new_name.push_str(count.to_string().as_str());
                new_name.finish()
            }
        }
    }

    /// Suggest a name for given type.
    ///
    /// The function will strip references first, and suggest name from the inner type.
    ///
    /// - If `ty` is an ADT, it will suggest the name of the ADT.
    ///   + If `ty` is wrapped in `Box`, `Option` or `Result`, it will suggest the name from the inner type.
    /// - If `ty` is a trait, it will suggest the name of the trait.
    /// - If `ty` is an `impl Trait`, it will suggest the name of the first trait.
    ///
    /// If the suggested name conflicts with reserved keywords, it will return `None`.
    pub fn for_type<'db>(
        &mut self,
        ty: &hir::Type<'db>,
        db: &'db RootDatabase,
        edition: Edition,
    ) -> Option<SmolStr> {
        let name = name_of_type(ty, db, edition)?;
        Some(self.suggest_name(&name))
    }

    /// Suggest name of impl trait type
    ///
    /// # Current implementation
    ///
    /// In current implementation, the function tries to get the name from the first
    /// character of the name for the first type bound.
    ///
    /// If the name conflicts with existing generic parameters, it will try to
    /// resolve the conflict with `for_unique_generic_name`.
    pub fn for_impl_trait_as_generic(&mut self, ty: &ast::ImplTraitType) -> SmolStr {
        let c = ty
            .type_bound_list()
            .and_then(|bounds| bounds.syntax().text().char_at(0.into()))
            .unwrap_or('T');

        self.suggest_name(&c.to_string())
    }

    /// Suggest name of variable for given expression
    ///
    /// In current implementation, the function tries to get the name from
    /// the following sources:
    ///
    /// * if expr is an argument to function/method, use parameter name
    /// * if expr is a function/method call, use function name
    /// * expression type name if it exists (E.g. `()`, `fn() -> ()` or `!` do not have names)
    /// * fallback: `var_name`
    ///
    /// It also applies heuristics to filter out less informative names
    ///
    /// Currently it sticks to the first name found.
    pub fn for_variable(
        &mut self,
        expr: &ast::Expr,
        sema: &Semantics<'_, RootDatabase>,
    ) -> SmolStr {
        // `from_param` does not benefit from stripping it need the largest
        // context possible so we check firstmost
        if let Some(name) = from_param(expr, sema) {
            return self.suggest_name(&name);
        }

        let mut next_expr = Some(expr.clone());
        while let Some(expr) = next_expr {
            let name = from_call(&expr)
                .or_else(|| from_type(&expr, sema))
                .or_else(|| from_field_name(&expr));
            if let Some(name) = name {
                return self.suggest_name(&name);
            }

            match expr {
                ast::Expr::RefExpr(inner) => next_expr = inner.expr(),
                ast::Expr::AwaitExpr(inner) => next_expr = inner.expr(),
                // ast::Expr::BlockExpr(block) => expr = block.tail_expr(),
                ast::Expr::CastExpr(inner) => next_expr = inner.expr(),
                ast::Expr::MethodCallExpr(method) if is_useless_method(&method) => {
                    next_expr = method.receiver();
                }
                ast::Expr::ParenExpr(inner) => next_expr = inner.expr(),
                ast::Expr::TryExpr(inner) => next_expr = inner.expr(),
                ast::Expr::PrefixExpr(prefix) if prefix.op_kind() == Some(ast::UnaryOp::Deref) => {
                    next_expr = prefix.expr()
                }
                _ => break,
            }
        }

        self.suggest_name("var_name")
    }

    /// Insert a name into the pool
    fn insert(&mut self, name: &str) {
        let (prefix, suffix) = Self::split_numeric_suffix(name);
        let prefix = SmolStr::new(prefix);
        let suffix = suffix.unwrap_or(0);

        match self.pool.entry(prefix) {
            Entry::Vacant(entry) => {
                entry.insert(suffix);
            }
            Entry::Occupied(mut entry) => {
                let count = entry.get_mut();
                *count = (*count).max(suffix);
            }
        }
    }

    /// Remove the numeric suffix from the name
    ///
    /// # Examples
    /// `a1b2c3` -> `a1b2c`
    fn split_numeric_suffix(name: &str) -> (&str, Option<usize>) {
        let pos =
            name.rfind(|c: char| !c.is_numeric()).expect("Name cannot be empty or all-numeric");
        let (prefix, suffix) = name.split_at(pos + 1);
        (prefix, suffix.parse().ok())
    }
}

fn normalize(name: &str) -> Option<SmolStr> {
    let name = to_lower_snake_case(name).to_smolstr();

    if USELESS_NAMES.contains(&name.as_str()) {
        return None;
    }

    if USELESS_NAME_PREFIXES.iter().any(|prefix| name.starts_with(prefix)) {
        return None;
    }

    if !is_valid_name(&name) {
        return None;
    }

    Some(name)
}

fn is_valid_name(name: &str) -> bool {
    matches!(
        super::LexedStr::single_token(syntax::Edition::CURRENT_FIXME, name),
        Some((syntax::SyntaxKind::IDENT, _error))
    )
}

fn is_useless_method(method: &ast::MethodCallExpr) -> bool {
    let ident = method.name_ref().and_then(|it| it.ident_token());

    match ident {
        Some(ident) => USELESS_METHODS.contains(&ident.text()),
        None => false,
    }
}

fn from_call(expr: &ast::Expr) -> Option<SmolStr> {
    from_func_call(expr).or_else(|| from_method_call(expr))
}

fn from_func_call(expr: &ast::Expr) -> Option<SmolStr> {
    let call = match expr {
        ast::Expr::CallExpr(call) => call,
        _ => return None,
    };
    let func = match call.expr()? {
        ast::Expr::PathExpr(path) => path,
        _ => return None,
    };
    let ident = func.path()?.segment()?.name_ref()?.ident_token()?;
    normalize(ident.text())
}

fn from_method_call(expr: &ast::Expr) -> Option<SmolStr> {
    let method = match expr {
        ast::Expr::MethodCallExpr(call) => call,
        _ => return None,
    };
    let ident = method.name_ref()?.ident_token()?;
    let mut name = ident.text();

    if USELESS_METHODS.contains(&name) {
        return None;
    }

    for prefix in USELESS_METHOD_PREFIXES {
        if let Some(suffix) = name.strip_prefix(prefix) {
            name = suffix;
            break;
        }
    }

    normalize(name)
}

fn from_param(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<SmolStr> {
    let arg_list = expr.syntax().parent().and_then(ast::ArgList::cast)?;
    let args_parent = arg_list.syntax().parent()?;
    let func = match_ast! {
        match args_parent {
            ast::CallExpr(call) => {
                let func = call.expr()?;
                let func_ty = sema.type_of_expr(&func)?.adjusted();
                func_ty.as_callable(sema.db)?
            },
            ast::MethodCallExpr(method) => sema.resolve_method_call_as_callable(&method)?,
            _ => return None,
        }
    };

    let (idx, _) = arg_list.args().find_position(|it| it == expr).unwrap();
    let param = func.params().into_iter().nth(idx)?;
    let pat = sema.source(param)?.value.right()?.pat()?;
    let name = var_name_from_pat(&pat)?;
    normalize(&name.to_smolstr())
}

fn var_name_from_pat(pat: &ast::Pat) -> Option<ast::Name> {
    match pat {
        ast::Pat::IdentPat(var) => var.name(),
        ast::Pat::RefPat(ref_pat) => var_name_from_pat(&ref_pat.pat()?),
        ast::Pat::BoxPat(box_pat) => var_name_from_pat(&box_pat.pat()?),
        _ => None,
    }
}

fn from_type(expr: &ast::Expr, sema: &Semantics<'_, RootDatabase>) -> Option<SmolStr> {
    let ty = sema.type_of_expr(expr)?.adjusted();
    let ty = ty.remove_ref().unwrap_or(ty);
    let edition = sema.scope(expr.syntax())?.krate().edition(sema.db);

    name_of_type(&ty, sema.db, edition)
}

fn name_of_type<'db>(
    ty: &hir::Type<'db>,
    db: &'db RootDatabase,
    edition: Edition,
) -> Option<SmolStr> {
    let name = if let Some(adt) = ty.as_adt() {
        let name = adt.name(db).display(db, edition).to_string();

        if WRAPPER_TYPES.contains(&name.as_str()) {
            let inner_ty = ty.type_arguments().next()?;
            return name_of_type(&inner_ty, db, edition);
        }

        if SEQUENCE_TYPES.contains(&name.as_str()) {
            let inner_ty = ty.type_arguments().next();
            return Some(sequence_name(inner_ty.as_ref(), db, edition));
        }

        name
    } else if let Some(trait_) = ty.as_dyn_trait() {
        trait_name(&trait_, db, edition)?
    } else if let Some(traits) = ty.as_impl_traits(db) {
        let mut iter = traits.filter_map(|t| trait_name(&t, db, edition));
        let name = iter.next()?;
        if iter.next().is_some() {
            return None;
        }
        name
    } else if let Some(inner_ty) = ty.remove_ref() {
        return name_of_type(&inner_ty, db, edition);
    } else if let Some(inner_ty) = ty.as_slice() {
        return Some(sequence_name(Some(&inner_ty), db, edition));
    } else {
        return None;
    };
    normalize(&name)
}

fn sequence_name<'db>(
    inner_ty: Option<&hir::Type<'db>>,
    db: &'db RootDatabase,
    edition: Edition,
) -> SmolStr {
    let items_str = SmolStr::new_static("items");
    let Some(inner_ty) = inner_ty else {
        return items_str;
    };
    let Some(name) = name_of_type(inner_ty, db, edition) else {
        return items_str;
    };

    if name.ends_with(['s', 'x', 'y']) {
        // Given a type called e.g. "Boss", "Fox" or "Story", don't try to
        // create a plural.
        items_str
    } else {
        SmolStr::new(format!("{name}s"))
    }
}

fn trait_name(trait_: &hir::Trait, db: &RootDatabase, edition: Edition) -> Option<String> {
    let name = trait_.name(db).display(db, edition).to_string();
    if USELESS_TRAITS.contains(&name.as_str()) {
        return None;
    }
    Some(name)
}

fn from_field_name(expr: &ast::Expr) -> Option<SmolStr> {
    let field = match expr {
        ast::Expr::FieldExpr(field) => field,
        _ => return None,
    };
    let ident = field.name_ref()?.ident_token()?;
    normalize(ident.text())
}

#[cfg(test)]
mod tests {
    use hir::FileRange;
    use test_fixture::WithFixture;

    use super::*;

    #[track_caller]
    fn check(#[rust_analyzer::rust_fixture] ra_fixture: &str, expected: &str) {
        let (db, file_id, range_or_offset) = RootDatabase::with_range_or_offset(ra_fixture);
        let frange = FileRange { file_id, range: range_or_offset.into() };
        let sema = Semantics::new(&db);

        let source_file = sema.parse(frange.file_id);

        let element = source_file.syntax().covering_element(frange.range);
        let expr =
            element.ancestors().find_map(ast::Expr::cast).expect("selection is not an expression");
        assert_eq!(
            expr.syntax().text_range(),
            frange.range,
            "selection is not an expression(yet contained in one)"
        );
        let name = NameGenerator::default().for_variable(&expr, &sema);
        assert_eq!(&name, expected);
    }

    #[test]
    fn no_args() {
        check(r#"fn foo() { $0bar()$0 }"#, "bar");
        check(r#"fn foo() { $0bar.frobnicate()$0 }"#, "frobnicate");
    }

    #[test]
    fn single_arg() {
        check(r#"fn foo() { $0bar(1)$0 }"#, "bar");
    }

    #[test]
    fn many_args() {
        check(r#"fn foo() { $0bar(1, 2, 3)$0 }"#, "bar");
    }

    #[test]
    fn path() {
        check(r#"fn foo() { $0i32::bar(1, 2, 3)$0 }"#, "bar");
    }

    #[test]
    fn generic_params() {
        check(r#"fn foo() { $0bar::<i32>(1, 2, 3)$0 }"#, "bar");
        check(r#"fn foo() { $0bar.frobnicate::<i32, u32>()$0 }"#, "frobnicate");
    }

    #[test]
    fn to_name() {
        check(
            r#"
struct Args;
struct Config;
impl Args {
    fn to_config(&self) -> Config {}
}
fn foo() {
    $0Args.to_config()$0;
}
"#,
            "config",
        );
    }

    #[test]
    fn plain_func() {
        check(
            r#"
fn bar(n: i32, m: u32);
fn foo() { bar($01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn mut_param() {
        check(
            r#"
fn bar(mut n: i32, m: u32);
fn foo() { bar($01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn func_does_not_exist() {
        check(r#"fn foo() { bar($01$0, 2) }"#, "var_name");
    }

    #[test]
    fn unnamed_param() {
        check(
            r#"
fn bar(_: i32, m: u32);
fn foo() { bar($01$0, 2) }
"#,
            "var_name",
        );
    }

    #[test]
    fn tuple_pat() {
        check(
            r#"
fn bar((n, k): (i32, i32), m: u32);
fn foo() {
    bar($0(1, 2)$0, 3)
}
"#,
            "var_name",
        );
    }

    #[test]
    fn ref_pat() {
        check(
            r#"
fn bar(&n: &i32, m: u32);
fn foo() { bar($0&1$0, 3) }
"#,
            "n",
        );
    }

    #[test]
    fn box_pat() {
        check(
            r#"
fn bar(box n: &i32, m: u32);
fn foo() { bar($01$0, 3) }
"#,
            "n",
        );
    }

    #[test]
    fn param_out_of_index() {
        check(
            r#"
fn bar(n: i32, m: u32);
fn foo() { bar(1, 2, $03$0) }
"#,
            "var_name",
        );
    }

    #[test]
    fn generic_param_resolved() {
        check(
            r#"
fn bar<T>(n: T, m: u32);
fn foo() { bar($01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn generic_param_unresolved() {
        check(
            r#"
fn bar<T>(n: T, m: u32);
fn foo<T>(x: T) { bar($0x$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn method() {
        check(
            r#"
struct S;
impl S { fn bar(&self, n: i32, m: u32); }
fn foo() { S.bar($01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn method_on_impl_trait() {
        check(
            r#"
struct S;
trait T {
    fn bar(&self, n: i32, m: u32);
}
impl T for S { fn bar(&self, n: i32, m: u32); }
fn foo() { S.bar($01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn method_ufcs() {
        check(
            r#"
struct S;
impl S { fn bar(&self, n: i32, m: u32); }
fn foo() { S::bar(&S, $01$0, 2) }
"#,
            "n",
        );
    }

    #[test]
    fn method_self() {
        check(
            r#"
struct S;
impl S { fn bar(&self, n: i32, m: u32); }
fn foo() { S::bar($0&S$0, 1, 2) }
"#,
            "s",
        );
    }

    #[test]
    fn method_self_named() {
        check(
            r#"
struct S;
impl S { fn bar(strukt: &Self, n: i32, m: u32); }
fn foo() { S::bar($0&S$0, 1, 2) }
"#,
            "strukt",
        );
    }

    #[test]
    fn i32() {
        check(r#"fn foo() { let _: i32 = $01$0; }"#, "var_name");
    }

    #[test]
    fn u64() {
        check(r#"fn foo() { let _: u64 = $01$0; }"#, "var_name");
    }

    #[test]
    fn bool() {
        check(r#"fn foo() { let _: bool = $0true$0; }"#, "var_name");
    }

    #[test]
    fn struct_unit() {
        check(
            r#"
struct Seed;
fn foo() { let _ = $0Seed$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn struct_unit_to_snake() {
        check(
            r#"
struct SeedState;
fn foo() { let _ = $0SeedState$0; }
"#,
            "seed_state",
        );
    }

    #[test]
    fn struct_single_arg() {
        check(
            r#"
struct Seed(u32);
fn foo() { let _ = $0Seed(0)$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn struct_with_fields() {
        check(
            r#"
struct Seed { value: u32 }
fn foo() { let _ = $0Seed { value: 0 }$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn enum_() {
        check(
            r#"
enum Kind { A, B }
fn foo() { let _ = $0Kind::A$0; }
"#,
            "kind",
        );
    }

    #[test]
    fn enum_generic_resolved() {
        check(
            r#"
enum Kind<T> { A { x: T }, B }
fn foo() { let _ = $0Kind::A { x:1 }$0; }
"#,
            "kind",
        );
    }

    #[test]
    fn enum_generic_unresolved() {
        check(
            r#"
enum Kind<T> { A { x: T }, B }
fn foo<T>(x: T) { let _ = $0Kind::A { x }$0; }
"#,
            "kind",
        );
    }

    #[test]
    fn dyn_trait() {
        check(
            r#"
trait DynHandler {}
fn bar() -> dyn DynHandler {}
fn foo() { $0(bar())$0; }
"#,
            "dyn_handler",
        );
    }

    #[test]
    fn impl_trait() {
        check(
            r#"
trait StaticHandler {}
fn bar() -> impl StaticHandler {}
fn foo() { $0(bar())$0; }
"#,
            "static_handler",
        );
    }

    #[test]
    fn impl_trait_plus_clone() {
        check(
            r#"
trait StaticHandler {}
trait Clone {}
fn bar() -> impl StaticHandler + Clone {}
fn foo() { $0(bar())$0; }
"#,
            "static_handler",
        );
    }

    #[test]
    fn impl_trait_plus_lifetime() {
        check(
            r#"
trait StaticHandler {}
trait Clone {}
fn bar<'a>(&'a i32) -> impl StaticHandler + 'a {}
fn foo() { $0(bar(&1))$0; }
"#,
            "static_handler",
        );
    }

    #[test]
    fn impl_trait_plus_trait() {
        check(
            r#"
trait Handler {}
trait StaticHandler {}
fn bar() -> impl StaticHandler + Handler {}
fn foo() { $0(bar())$0; }
"#,
            "bar",
        );
    }

    #[test]
    fn ref_value() {
        check(
            r#"
struct Seed;
fn bar() -> &Seed {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn box_value() {
        check(
            r#"
struct Box<T>(*const T);
struct Seed;
fn bar() -> Box<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn box_generic() {
        check(
            r#"
struct Box<T>(*const T);
fn bar<T>() -> Box<T> {}
fn foo<T>() { $0(bar::<T>())$0; }
"#,
            "bar",
        );
    }

    #[test]
    fn option_value() {
        check(
            r#"
enum Option<T> { Some(T) }
struct Seed;
fn bar() -> Option<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn result_value() {
        check(
            r#"
enum Result<T, E> { Ok(T), Err(E) }
struct Seed;
struct Error;
fn bar() -> Result<Seed, Error> {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn arc_value() {
        check(
            r#"
struct Arc<T>(*const T);
struct Seed;
fn bar() -> Arc<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn rc_value() {
        check(
            r#"
struct Rc<T>(*const T);
struct Seed;
fn bar() -> Rc<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seed",
        );
    }

    #[test]
    fn vec_value() {
        check(
            r#"
struct Vec<T> {};
struct Seed;
fn bar() -> Vec<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seeds",
        );
    }

    #[test]
    fn vec_value_ends_with_s() {
        check(
            r#"
struct Vec<T> {};
struct Boss;
fn bar() -> Vec<Boss> {}
fn foo() { $0(bar())$0; }
"#,
            "items",
        );
    }

    #[test]
    fn vecdeque_value() {
        check(
            r#"
struct VecDeque<T> {};
struct Seed;
fn bar() -> VecDeque<Seed> {}
fn foo() { $0(bar())$0; }
"#,
            "seeds",
        );
    }

    #[test]
    fn slice_value() {
        check(
            r#"
struct Vec<T> {};
struct Seed;
fn bar() -> &[Seed] {}
fn foo() { $0(bar())$0; }
"#,
            "seeds",
        );
    }

    #[test]
    fn ref_call() {
        check(
            r#"
fn foo() { $0&bar(1, 3)$0 }
"#,
            "bar",
        );
    }

    #[test]
    fn name_to_string() {
        check(
            r#"
fn foo() { $0function.name().to_string()$0 }
"#,
            "name",
        );
    }

    #[test]
    fn nested_useless_method() {
        check(
            r#"
fn foo() { $0function.name().as_ref().unwrap().to_string()$0 }
"#,
            "name",
        );
    }

    #[test]
    fn struct_field_name() {
        check(
            r#"
struct S<T> {
    some_field: T;
}
fn foo<T>(some_struct: S<T>) { $0some_struct.some_field$0 }
"#,
            "some_field",
        );
    }

    #[test]
    fn from_and_to_func() {
        check(
            r#"
//- minicore: from
struct Foo;
struct Bar;

impl From<Foo> for Bar {
    fn from(_: Foo) -> Self {
        Bar;
    }
}

fn f(_: Bar) {}

fn main() {
    let foo = Foo {};
    f($0Bar::from(foo)$0);
}
"#,
            "bar",
        );

        check(
            r#"
//- minicore: from
struct Foo;
struct Bar;

impl From<Foo> for Bar {
    fn from(_: Foo) -> Self {
        Bar;
    }
}

fn f(_: Bar) {}

fn main() {
    let foo = Foo {};
    f($0Into::<Bar>::into(foo)$0);
}
"#,
            "bar",
        );
    }

    #[test]
    fn useless_name_prefix() {
        check(
            r#"
struct Foo;
struct Bar;

impl Bar {
    fn from_foo(_: Foo) -> Self {
        Foo {}
    }
}

fn main() {
    let foo = Foo {};
    let _ = $0Bar::from_foo(foo)$0;
}
"#,
            "bar",
        );

        check(
            r#"
struct Foo;
struct Bar;

impl Bar {
    fn with_foo(_: Foo) -> Self {
        Bar {}
    }
}

fn main() {
    let foo = Foo {};
    let _ = $0Bar::with_foo(foo)$0;
}
"#,
            "bar",
        );
    }

    #[test]
    fn conflicts_with_existing_names() {
        let mut generator = NameGenerator::default();
        assert_eq!(generator.suggest_name("a"), "a");
        assert_eq!(generator.suggest_name("a"), "a1");
        assert_eq!(generator.suggest_name("a"), "a2");
        assert_eq!(generator.suggest_name("a"), "a3");

        assert_eq!(generator.suggest_name("b"), "b");
        assert_eq!(generator.suggest_name("b2"), "b2");
        assert_eq!(generator.suggest_name("b"), "b3");
        assert_eq!(generator.suggest_name("b"), "b4");
        assert_eq!(generator.suggest_name("b3"), "b5");

        // ---------
        let mut generator = NameGenerator::new_with_names(["a", "b", "b2", "c4"].into_iter());
        assert_eq!(generator.suggest_name("a"), "a1");
        assert_eq!(generator.suggest_name("a"), "a2");

        assert_eq!(generator.suggest_name("b"), "b3");
        assert_eq!(generator.suggest_name("b2"), "b4");

        assert_eq!(generator.suggest_name("c"), "c5");
    }
}

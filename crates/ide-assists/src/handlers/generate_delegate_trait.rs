use std::ops::Not;

use crate::{
    assist_context::{AssistContext, Assists},
    utils::convert_param_list_to_arg_list,
};
use either::Either;
use hir::{db::HirDatabase, HasVisibility};
use ide_db::{
    assists::{AssistId, GroupLabel},
    path_transform::PathTransform,
};
use syntax::{
    ast::{
        self,
        edit::{self, AstNodeEdit},
        make, AssocItem, HasGenericParams, HasName, HasVisibility as astHasVisibility, Path,
    },
    ted::{self, Position},
    AstNode, NodeOrToken, SyntaxKind,
};

// Assist: generate_delegate_trait
//
// Generate delegate trait implementation for `StructField`s.
//
// ```
// trait SomeTrait {
//     type T;
//     fn fn_(arg: u32) -> u32;
//     fn method_(&mut self) -> bool;
// }
// struct A;
// impl SomeTrait for A {
//     type T = u32;
//
//     fn fn_(arg: u32) -> u32 {
//         42
//     }
//
//     fn method_(&mut self) -> bool {
//         false
//     }
// }
// struct B {
//     a$0: A,
// }
// ```
// ->
// ```
// trait SomeTrait {
//     type T;
//     fn fn_(arg: u32) -> u32;
//     fn method_(&mut self) -> bool;
// }
// struct A;
// impl SomeTrait for A {
//     type T = u32;
//
//     fn fn_(arg: u32) -> u32 {
//         42
//     }
//
//     fn method_(&mut self) -> bool {
//         false
//     }
// }
// struct B {
//     a: A,
// }
//
// impl SomeTrait for B {
//     type T = <A as SomeTrait>::T;
//
//     fn fn_(arg: u32) -> u32 {
//         <A as SomeTrait>::fn_(arg)
//     }
//
//     fn method_(&mut self) -> bool {
//         <A as SomeTrait>::method_( &mut self.a )
//     }
// }
// ```
pub(crate) fn generate_delegate_trait(acc: &mut Assists, ctx: &AssistContext<'_>) -> Option<()> {
    let strukt = Struct::new(ctx.find_node_at_offset::<ast::Struct>()?)?;

    let field: Field = match ctx.find_node_at_offset::<ast::RecordField>() {
        Some(field) => Field::new(&ctx, Either::Left(field))?,
        None => {
            let field = ctx.find_node_at_offset::<ast::TupleField>()?;
            let field_list = ctx.find_node_at_offset::<ast::TupleFieldList>()?;
            Field::new(&ctx, either::Right((field, field_list)))?
        }
    };

    strukt.delegate(field, acc, ctx);
    Some(())
}

/// A utility object that represents a struct's field.
struct Field {
    name: String,
    ty: ast::Type,
    range: syntax::TextRange,
    impls: Vec<Delegee>,
}

impl Field {
    pub(crate) fn new(
        ctx: &AssistContext<'_>,
        f: Either<ast::RecordField, (ast::TupleField, ast::TupleFieldList)>,
    ) -> Option<Field> {
        let db = ctx.sema.db;
        let name: String;
        let range: syntax::TextRange;
        let ty: ast::Type;

        let module = ctx.sema.to_module_def(ctx.file_id())?;

        match f {
            Either::Left(f) => {
                name = f.name()?.to_string();
                ty = f.ty()?;
                range = f.syntax().text_range();
            }
            Either::Right((f, l)) => {
                name = l.fields().position(|it| it == f)?.to_string();
                ty = f.ty()?;
                range = f.syntax().text_range();
            }
        };

        let hir_ty = ctx.sema.resolve_type(&ty)?;
        let type_impls = hir::Impl::all_for_type(db, hir_ty.clone());
        let mut impls = Vec::with_capacity(type_impls.len());
        let type_param = hir_ty.as_type_param(db);

        if let Some(tp) = type_param {
            for tb in tp.trait_bounds(db) {
                impls.push(Delegee::Bound(BoundCase(tb)));
            }
        };

        for imp in type_impls {
            match imp.trait_(db) {
                Some(tr) => {
                    if tr.is_visible_from(db, module) {
                        impls.push(Delegee::Impls(ImplCase(tr, imp)))
                    }
                }
                None => (),
            }
        }

        Some(Field { name, ty, range, impls })
    }
}

/// A field that we want to delegate can offer the enclosing struct
/// trait to implement in two ways. The first way is when the field
/// actually implements the trait and the second way is when the field
/// has a bound type parameter. We handle these cases in different ways
/// hence the enum.
enum Delegee {
    Bound(BoundCase),
    Impls(ImplCase),
}

struct BoundCase(hir::Trait);
struct ImplCase(hir::Trait, hir::Impl);

impl Delegee {
    fn signature(&self, db: &dyn HirDatabase) -> String {
        let mut s = String::new();

        let (Delegee::Bound(BoundCase(it)) | Delegee::Impls(ImplCase(it, _))) = self;

        for m in it.module(db).path_to_root(db).iter().rev() {
            if let Some(name) = m.name(db) {
                s.push_str(&format!("{}::", name.to_smol_str()));
            }
        }

        s.push_str(&it.name(db).to_smol_str());
        s
    }
}

/// A utility struct that is used for the enclosing struct.
struct Struct {
    strukt: ast::Struct,
    name: ast::Name,
}

impl Struct {
    pub(crate) fn new(s: ast::Struct) -> Option<Self> {
        let name = s.name()?;
        Some(Struct { name, strukt: s })
    }

    pub(crate) fn delegate(&self, field: Field, acc: &mut Assists, ctx: &AssistContext<'_>) {
        let db = ctx.db();
        for delegee in &field.impls {
            // FIXME :  We can omit already implemented impl_traits
            // But we don't know what the &[hir::Type] argument should look like.

            // let trait_ = match delegee {
            //     Delegee::Bound(b) => b.0,
            //     Delegee::Impls(i) => i.1,
            // };

            // if self.hir_ty.impls_trait(db, trait_, &[]) {
            //     continue;
            // }
            let signature = delegee.signature(db);
            let delegate = generate_impl(ctx, self, &field.ty, &field.name, delegee);

            acc.add_group(
                &GroupLabel("Delegate trait impl for field...".to_owned()),
                AssistId("generate_delegate_trait", ide_db::assists::AssistKind::Generate),
                format!("Generate delegate impl `{}` for `{}`", signature, field.name),
                field.range,
                |builder| {
                    builder.insert(
                        self.strukt.syntax().text_range().end(),
                        format!("\n\n{}", delegate.syntax()),
                    );
                },
            );
        }
    }
}

fn generate_impl(
    ctx: &AssistContext<'_>,
    strukt: &Struct,
    field_ty: &ast::Type,
    field_name: &String,
    delegee: &Delegee,
) -> ast::Impl {
    let delegate: ast::Impl;
    let source: ast::Impl;
    let genpar: Option<ast::GenericParamList>;
    let db = ctx.db();
    let base_path = make::path_from_text(&field_ty.to_string().as_str());
    let s_path = make::ext::ident_path(&strukt.name.to_string());

    match delegee {
        Delegee::Bound(delegee) => {
            let in_file = ctx.sema.source(delegee.0.to_owned()).unwrap();
            let source: ast::Trait = in_file.value;

            delegate = make::impl_trait(
                delegee.0.is_unsafe(db),
                None,
                None,
                strukt.strukt.generic_param_list(),
                None,
                delegee.0.is_auto(db),
                make::ty(&delegee.0.name(db).to_smol_str()),
                make::ty_path(s_path),
                source.where_clause(),
                strukt.strukt.where_clause(),
                None,
            )
            .clone_for_update();

            genpar = source.generic_param_list();
            let delegate_assoc_items = delegate.get_or_create_assoc_item_list();
            let gen_args: String =
                genpar.map_or_else(String::new, |params| params.to_generic_args().to_string());

            // Goto link : https://doc.rust-lang.org/reference/paths.html#qualified-paths
            let qualified_path_type = make::path_from_text(&format!(
                "<{} as {}{}>",
                base_path.to_string(),
                delegee.0.name(db).to_smol_str(),
                gen_args.to_string()
            ));

            match source.assoc_item_list() {
                Some(ai) => {
                    ai.assoc_items()
                        .filter(|item| matches!(item, AssocItem::MacroCall(_)).not())
                        .for_each(|item| {
                            let assoc =
                                process_assoc_item(item, qualified_path_type.clone(), &field_name);
                            if let Some(assoc) = assoc {
                                delegate_assoc_items.add_item(assoc);
                            }
                        });
                }
                None => {}
            };

            let target = ctx.sema.scope(strukt.strukt.syntax()).unwrap();
            let source = ctx.sema.scope(source.syntax()).unwrap();

            let transform =
                PathTransform::trait_impl(&target, &source, delegee.0, delegate.clone());
            transform.apply(&delegate.syntax());
        }
        Delegee::Impls(delegee) => {
            let in_file = ctx.sema.source(delegee.1.to_owned()).unwrap();
            source = in_file.value;
            delegate = make::impl_trait(
                delegee.0.is_unsafe(db),
                source.generic_param_list(),
                None,
                None,
                None,
                delegee.0.is_auto(db),
                make::ty(&delegee.0.name(db).to_smol_str()),
                make::ty_path(s_path),
                source.where_clause(),
                strukt.strukt.where_clause(),
                None,
            )
            .clone_for_update();
            genpar = source.generic_param_list();
            let delegate_assoc_items = delegate.get_or_create_assoc_item_list();
            let gen_args: String =
                genpar.map_or_else(String::new, |params| params.to_generic_args().to_string());

            // Goto link : https://doc.rust-lang.org/reference/paths.html#qualified-paths
            let qualified_path_type = make::path_from_text(&format!(
                "<{} as {}{}>",
                base_path.to_string().as_str(),
                delegee.0.name(db).to_smol_str(),
                gen_args.to_string().as_str()
            ));

            source
                .get_or_create_assoc_item_list()
                .assoc_items()
                .filter(|item| matches!(item, AssocItem::MacroCall(_)).not())
                .for_each(|item| {
                    let assoc = process_assoc_item(item, qualified_path_type.clone(), &field_name);
                    if let Some(assoc) = assoc {
                        delegate_assoc_items.add_item(assoc);
                    }
                });

            let target = ctx.sema.scope(strukt.strukt.syntax()).unwrap();
            let source = ctx.sema.scope(source.syntax()).unwrap();

            let transform =
                PathTransform::trait_impl(&target, &source, delegee.0, delegate.clone());
            transform.apply(&delegate.syntax());
        }
    }

    delegate
}

fn process_assoc_item(
    item: syntax::ast::AssocItem,
    qual_path_ty: ast::Path,
    base_name: &str,
) -> Option<ast::AssocItem> {
    match item {
        AssocItem::Const(c) => Some(const_assoc_item(c, qual_path_ty)),
        AssocItem::Fn(f) => Some(func_assoc_item(f, qual_path_ty, base_name)),
        AssocItem::MacroCall(_) => {
            // FIXME : Handle MacroCall case.
            // return Some(macro_assoc_item(mac, qual_path_ty));
            None
        }
        AssocItem::TypeAlias(ta) => Some(ty_assoc_item(ta, qual_path_ty)),
    }
}

fn const_assoc_item(item: syntax::ast::Const, qual_path_ty: ast::Path) -> AssocItem {
    let path_expr_segment = make::path_from_text(item.name().unwrap().to_string().as_str());

    // We want rhs of the const assignment to be a qualified path
    // The general case for const assigment can be found [here](`https://doc.rust-lang.org/reference/items/constant-items.html`)
    // The qualified will have the following generic syntax :
    // <Base as Trait<GenArgs>>::ConstName;
    // FIXME : We can't rely on `make::path_qualified` for now but it would be nice to replace the following with it.
    // make::path_qualified(qual_path_ty, path_expr_segment.as_single_segment().unwrap());
    let qualpath = qualpath(qual_path_ty, path_expr_segment);
    let inner = make::item_const(
        item.visibility(),
        item.name().unwrap(),
        item.ty().unwrap(),
        make::expr_path(qualpath),
    )
    .clone_for_update();

    AssocItem::Const(inner)
}

fn func_assoc_item(item: syntax::ast::Fn, qual_path_ty: Path, base_name: &str) -> AssocItem {
    let path_expr_segment = make::path_from_text(item.name().unwrap().to_string().as_str());
    let qualpath = qualpath(qual_path_ty, path_expr_segment);

    let call = match item.param_list() {
        // Methods and funcs should be handled separately.
        // We ask if the func has a `self` param.
        Some(l) => match l.self_param() {
            Some(slf) => {
                let mut self_kw = make::expr_path(make::path_from_text("self"));
                self_kw = make::expr_field(self_kw, base_name);

                let tail_expr_self = match slf.kind() {
                    ast::SelfParamKind::Owned => self_kw,
                    ast::SelfParamKind::Ref => make::expr_ref(self_kw, false),
                    ast::SelfParamKind::MutRef => make::expr_ref(self_kw, true),
                };

                let param_count = l.params().count();
                let args = convert_param_list_to_arg_list(l).clone_for_update();

                if param_count > 0 {
                    // Add SelfParam and a TOKEN::COMMA
                    ted::insert_all(
                        Position::after(args.l_paren_token().unwrap()),
                        vec![
                            NodeOrToken::Node(tail_expr_self.syntax().clone_for_update()),
                            NodeOrToken::Token(make::token(SyntaxKind::WHITESPACE)),
                            NodeOrToken::Token(make::token(SyntaxKind::COMMA)),
                        ],
                    );
                } else {
                    // Add SelfParam only
                    ted::insert(
                        Position::after(args.l_paren_token().unwrap()),
                        NodeOrToken::Node(tail_expr_self.syntax().clone_for_update()),
                    );
                }

                make::expr_call(make::expr_path(qualpath), args)
            }
            None => make::expr_call(make::expr_path(qualpath), convert_param_list_to_arg_list(l)),
        },
        None => make::expr_call(
            make::expr_path(qualpath),
            convert_param_list_to_arg_list(make::param_list(None, Vec::new())),
        ),
    }
    .clone_for_update();

    let body = make::block_expr(vec![], Some(call)).clone_for_update();
    let func = make::fn_(
        item.visibility(),
        item.name().unwrap(),
        item.generic_param_list(),
        item.where_clause(),
        item.param_list().unwrap(),
        body,
        item.ret_type(),
        item.async_token().is_some(),
        item.const_token().is_some(),
        item.unsafe_token().is_some(),
    )
    .clone_for_update();

    AssocItem::Fn(func.indent(edit::IndentLevel(1)).clone_for_update())
}

fn ty_assoc_item(item: syntax::ast::TypeAlias, qual_path_ty: Path) -> AssocItem {
    let path_expr_segment = make::path_from_text(item.name().unwrap().to_string().as_str());
    let qualpath = qualpath(qual_path_ty, path_expr_segment);
    let ty = make::ty_path(qualpath);
    let ident = item.name().unwrap().to_string();

    let alias = make::ty_alias(
        ident.as_str(),
        item.generic_param_list(),
        None,
        item.where_clause(),
        Some((ty, None)),
    )
    .clone_for_update();

    AssocItem::TypeAlias(alias)
}

fn qualpath(qual_path_ty: ast::Path, path_expr_seg: ast::Path) -> ast::Path {
    make::path_from_text(&format!("{}::{}", qual_path_ty.to_string(), path_expr_seg.to_string()))
}

#[cfg(test)]
mod test {

    use super::*;
    use crate::tests::{check_assist, check_assist_not_applicable};

    #[test]
    fn test_tuple_struct_basic() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S(B$0ase);
trait Trait {}
impl Trait for Base {}
"#,
            r#"
struct Base;
struct S(Base);

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_struct_struct_basic() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se : Base
}
trait Trait {}
impl Trait for Base {}
"#,
            r#"
struct Base;
struct S {
    base : Base
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        )
    }

    // Structs need to be by def populated with fields
    // However user can invoke this assist while still editing
    // We therefore assert its non-applicability
    #[test]
    fn test_yet_empty_struct() {
        check_assist_not_applicable(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    $0
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        )
    }

    #[test]
    fn test_yet_unspecified_field_type() {
        check_assist_not_applicable(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ab$0c
}

impl Trait for S {}
trait Trait {}
impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_unsafe_trait() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se : Base
}
unsafe trait Trait {}
unsafe impl Trait for Base {}
"#,
            r#"
struct Base;
struct S {
    base : Base
}

unsafe impl Trait for S {}
unsafe trait Trait {}
unsafe impl Trait for Base {}
"#,
        );
    }

    #[test]
    fn test_unsafe_trait_with_unsafe_fn() {
        check_assist(
            generate_delegate_trait,
            r#"
struct Base;
struct S {
    ba$0se: Base,
}

unsafe trait Trait {
    unsafe fn a_func();
    unsafe fn a_method(&self);
}
unsafe impl Trait for Base {
    unsafe fn a_func() {}
    unsafe fn a_method(&self) {}
}
"#,
            r#"
struct Base;
struct S {
    base: Base,
}

unsafe impl Trait for S {
    unsafe fn a_func() {
        <Base as Trait>::a_func()
    }

    unsafe fn a_method(&self) {
        <Base as Trait>::a_method( &self.base )
    }
}

unsafe trait Trait {
    unsafe fn a_func();
    unsafe fn a_method(&self);
}
unsafe impl Trait for Base {
    unsafe fn a_func() {}
    unsafe fn a_method(&self) {}
}
"#,
        );
    }

    #[test]
    fn test_struct_with_where_clause() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b$0 : T,
}"#,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b : T,
}

impl<T> AnotherTrait for S<T>
where
    T: AnotherTrait,
{}"#,
        );
    }

    #[test]
    fn test_complex_without_where() {
        check_assist(
            generate_delegate_trait,
            r#"
trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field$0: Base
}

impl<'a, T, const C: usize> Trait<'a, T, C> for Base {
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}
"#,
            r#"
trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base
}

impl<'a, T, const C: usize> Trait<'a, T, C> for S {
    type AssocType = <Base as Trait<'a, T, C>>::AssocType;

    const AssocConst: usize = <Base as Trait<'a, T, C>>::AssocConst;

    fn assoc_fn(p: ()) {
        <Base as Trait<'a, T, C>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'a, T, C>>::assoc_method( &self.field , p)
    }
}

impl<'a, T, const C: usize> Trait<'a, T, C> for Base {
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}
"#,
        );
    }

    #[test]
    fn test_complex_two() {
        check_assist(
            generate_delegate_trait,
            r"
trait AnotherTrait {}

trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    fi$0eld: Base,
}

impl<'b, C, const D: usize> Trait<'b, C, D> for Base
where
    C: AnotherTrait,
{
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}",
            r#"
trait AnotherTrait {}

trait Trait<'a, T, const C: usize> {
    type AssocType;
    const AssocConst: usize;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base,
}

impl<'b, C, const D: usize> Trait<'b, C, D> for S
where
    C: AnotherTrait,
{
    type AssocType = <Base as Trait<'b, C, D>>::AssocType;

    const AssocConst: usize = <Base as Trait<'b, C, D>>::AssocConst;

    fn assoc_fn(p: ()) {
        <Base as Trait<'b, C, D>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'b, C, D>>::assoc_method( &self.field , p)
    }
}

impl<'b, C, const D: usize> Trait<'b, C, D> for Base
where
    C: AnotherTrait,
{
    type AssocType = ();
    const AssocConst: usize = 0;
    fn assoc_fn(p: ()) {}
    fn assoc_method(&self, p: ()) {}
}"#,
        )
    }

    #[test]
    fn test_complex_three() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
trait YetAnotherTrait {}

struct StructImplsAll();
impl AnotherTrait for StructImplsAll {}
impl YetAnotherTrait for StructImplsAll {}

trait Trait<'a, T, const C: usize> {
    type A;
    const ASSOC_CONST: usize = C;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    fi$0eld: Base,
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for Base
where
    A: AnotherTrait,
{
    type A = i32;

    const ASSOC_CONST: usize = B;

    fn assoc_fn(p: ()) {}

    fn assoc_method(&self, p: ()) {}
}
"#,
            r#"
trait AnotherTrait {}
trait YetAnotherTrait {}

struct StructImplsAll();
impl AnotherTrait for StructImplsAll {}
impl YetAnotherTrait for StructImplsAll {}

trait Trait<'a, T, const C: usize> {
    type A;
    const ASSOC_CONST: usize = C;
    fn assoc_fn(p: ());
    fn assoc_method(&self, p: ());
}

struct Base;
struct S {
    field: Base,
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for S
where
    A: AnotherTrait,
{
    type A = <Base as Trait<'b, A, B>>::A;

    const ASSOC_CONST: usize = <Base as Trait<'b, A, B>>::ASSOC_CONST;

    fn assoc_fn(p: ()) {
        <Base as Trait<'b, A, B>>::assoc_fn(p)
    }

    fn assoc_method(&self, p: ()) {
        <Base as Trait<'b, A, B>>::assoc_method( &self.field , p)
    }
}

impl<'b, A: AnotherTrait + YetAnotherTrait, const B: usize> Trait<'b, A, B> for Base
where
    A: AnotherTrait,
{
    type A = i32;

    const ASSOC_CONST: usize = B;

    fn assoc_fn(p: ()) {}

    fn assoc_method(&self, p: ()) {}
}
"#,
        )
    }

    #[test]
    fn test_type_bound() {
        check_assist(
            generate_delegate_trait,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b$0: T,
}"#,
            r#"
trait AnotherTrait {}
struct S<T>
where
    T: AnotherTrait,
{
    b: T,
}

impl<T> AnotherTrait for S<T>
where
    T: AnotherTrait,
{}"#,
        );
    }

    #[test]
    fn test_docstring_example() {
        check_assist(
            generate_delegate_trait,
            r#"
trait SomeTrait {
    type T;
    fn fn_(arg: u32) -> u32;
    fn method_(&mut self) -> bool;
}
struct A;
impl SomeTrait for A {
    type T = u32;
    fn fn_(arg: u32) -> u32 {
        42
    }
    fn method_(&mut self) -> bool {
        false
    }
}
struct B {
    a$0: A,
}
"#,
            r#"
trait SomeTrait {
    type T;
    fn fn_(arg: u32) -> u32;
    fn method_(&mut self) -> bool;
}
struct A;
impl SomeTrait for A {
    type T = u32;
    fn fn_(arg: u32) -> u32 {
        42
    }
    fn method_(&mut self) -> bool {
        false
    }
}
struct B {
    a: A,
}

impl SomeTrait for B {
    type T = <A as SomeTrait>::T;

    fn fn_(arg: u32) -> u32 {
        <A as SomeTrait>::fn_(arg)
    }

    fn method_(&mut self) -> bool {
        <A as SomeTrait>::method_( &mut self.a )
    }
}
"#,
        );
    }

    #[test]
    fn import_from_other_mod() {
        check_assist(
            generate_delegate_trait,
            r#"
mod some_module {
    pub trait SomeTrait {
        type T;
        fn fn_(arg: u32) -> u32;
        fn method_(&mut self) -> bool;
    }
    pub struct A;
    impl SomeTrait for A {
        type T = u32;

        fn fn_(arg: u32) -> u32 {
            42
        }

        fn method_(&mut self) -> bool {
            false
        }
    }
}

struct B {
    a$0: some_module::A,
}"#,
            r#"
mod some_module {
    pub trait SomeTrait {
        type T;
        fn fn_(arg: u32) -> u32;
        fn method_(&mut self) -> bool;
    }
    pub struct A;
    impl SomeTrait for A {
        type T = u32;

        fn fn_(arg: u32) -> u32 {
            42
        }

        fn method_(&mut self) -> bool {
            false
        }
    }
}

struct B {
    a: some_module::A,
}

impl some_module::SomeTrait for B {
    type T = <some_module::A as some_module::SomeTrait>::T;

    fn fn_(arg: u32) -> u32 {
        <some_module::A as some_module::SomeTrait>::fn_(arg)
    }

    fn method_(&mut self) -> bool {
        <some_module::A as some_module::SomeTrait>::method_( &mut self.a )
    }
}"#,
        )
    }
}

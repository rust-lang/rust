use super::{ItemTree, ModItem, ModKind};
use crate::{db::DefDatabase, test_db::TestDB};
use hir_expand::{db::AstDatabase, HirFileId, InFile};
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;
use ra_syntax::{ast, AstNode};
use rustc_hash::FxHashSet;
use std::sync::Arc;
use stdx::format_to;

fn test_inner_items(ra_fixture: &str) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let file_id = HirFileId::from(file_id);
    let tree = db.item_tree(file_id);
    let root = db.parse_or_expand(file_id).unwrap();
    let ast_id_map = db.ast_id_map(file_id);

    // Traverse the item tree and collect all module/impl/trait-level items as AST nodes.
    let mut outer_items = FxHashSet::default();
    let mut worklist = tree.top_level_items().to_vec();
    while let Some(item) = worklist.pop() {
        let node: ast::ModuleItem = match item {
            ModItem::Import(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::ExternCrate(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Function(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Struct(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Union(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Enum(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Const(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Static(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::TypeAlias(it) => tree.source(&db, InFile::new(file_id, it)).into(),
            ModItem::Mod(it) => {
                if let ModKind::Inline { items } = &tree[it].kind {
                    worklist.extend(&**items);
                }
                tree.source(&db, InFile::new(file_id, it)).into()
            }
            ModItem::Trait(it) => {
                worklist.extend(tree[it].items.iter().map(|item| ModItem::from(*item)));
                tree.source(&db, InFile::new(file_id, it)).into()
            }
            ModItem::Impl(it) => {
                worklist.extend(tree[it].items.iter().map(|item| ModItem::from(*item)));
                tree.source(&db, InFile::new(file_id, it)).into()
            }
            ModItem::MacroCall(_) => continue,
        };

        outer_items.insert(node);
    }

    // Now descend the root node and check that all `ast::ModuleItem`s are either recorded above, or
    // registered as inner items.
    for item in root.descendants().skip(1).filter_map(ast::ModuleItem::cast) {
        if outer_items.contains(&item) {
            continue;
        }

        let ast_id = ast_id_map.ast_id(&item);
        assert!(!tree.inner_items(ast_id).is_empty());
    }
}

fn item_tree(ra_fixture: &str) -> Arc<ItemTree> {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    db.item_tree(file_id.into())
}

fn print_item_tree(ra_fixture: &str) -> String {
    let tree = item_tree(ra_fixture);
    let mut out = String::new();

    format_to!(out, "inner attrs: {:?}\n\n", tree.top_level_attrs());
    format_to!(out, "top-level items:\n");
    for item in tree.top_level_items() {
        fmt_mod_item(&mut out, &tree, *item);
        format_to!(out, "\n");
    }

    if !tree.inner_items.is_empty() {
        format_to!(out, "\ninner items:\n\n");
        for (ast_id, items) in &tree.inner_items {
            format_to!(out, "for AST {:?}:\n", ast_id);
            for inner in items {
                fmt_mod_item(&mut out, &tree, *inner);
                format_to!(out, "\n\n");
            }
        }
    }

    out
}

fn fmt_mod_item(out: &mut String, tree: &ItemTree, item: ModItem) {
    let attrs = tree.attrs(item.into());
    if !attrs.is_empty() {
        format_to!(out, "#[{:?}]\n", attrs);
    }

    let mut children = String::new();
    match item {
        ModItem::ExternCrate(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Import(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Function(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Struct(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Union(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Enum(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Const(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Static(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Trait(it) => {
            format_to!(out, "{:?}", tree[it]);
            for item in &*tree[it].items {
                fmt_mod_item(&mut children, tree, ModItem::from(*item));
                format_to!(children, "\n");
            }
        }
        ModItem::Impl(it) => {
            format_to!(out, "{:?}", tree[it]);
            for item in &*tree[it].items {
                fmt_mod_item(&mut children, tree, ModItem::from(*item));
                format_to!(children, "\n");
            }
        }
        ModItem::TypeAlias(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Mod(it) => {
            format_to!(out, "{:?}", tree[it]);
            match &tree[it].kind {
                ModKind::Inline { items } => {
                    for item in &**items {
                        fmt_mod_item(&mut children, tree, *item);
                        format_to!(children, "\n");
                    }
                }
                ModKind::Outline {} => {}
            }
        }
        ModItem::MacroCall(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
    }

    for line in children.lines() {
        format_to!(out, "\n> {}", line);
    }
}

#[test]
fn smoke() {
    assert_snapshot!(print_item_tree(r"
        #![attr]

        #[attr_on_use]
        use {a, b::*};

        #[ext_crate]
        extern crate krate;

        #[on_trait]
        trait Tr<U> {
            #[assoc_ty]
            type AssocTy: Tr<()>;

            #[assoc_const]
            const CONST: u8;

            #[assoc_method]
            fn method(&self);

            #[assoc_dfl_method]
            fn dfl_method(&mut self) {}
        }

        #[struct0]
        struct Struct0<T = ()>;

        #[struct1]
        struct Struct1<T>(#[struct1fld] u8);

        #[struct2]
        struct Struct2<T> {
            #[struct2fld]
            fld: (T, ),
        }

        #[en]
        enum En {
            #[enum_variant]
            Variant {
                #[enum_field]
                field: u8,
            },
        }

        #[un]
        union Un {
            #[union_fld]
            fld: u16,
        }
    "), @r###"
inner attrs: Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr"))] }, input: None }]) }

top-level items:
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_on_use"))] }, input: None }]) }]
Import { path: ModPath { kind: Plain, segments: [Name(Text("a"))] }, alias: None, visibility: RawVisibilityId("pub(self)"), is_glob: false, is_prelude: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UseItem>(0) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_on_use"))] }, input: None }]) }]
Import { path: ModPath { kind: Plain, segments: [Name(Text("b"))] }, alias: None, visibility: RawVisibilityId("pub(self)"), is_glob: true, is_prelude: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UseItem>(0) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("ext_crate"))] }, input: None }]) }]
ExternCrate { path: ModPath { kind: Plain, segments: [Name(Text("krate"))] }, alias: None, visibility: RawVisibilityId("pub(self)"), is_macro_use: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ExternCrateItem>(1) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("on_trait"))] }, input: None }]) }]
Trait { name: Name(Text("Tr")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(0), auto: false, items: [TypeAlias(Idx::<TypeAlias>(0)), Const(Idx::<Const>(0)), Function(Idx::<Function>(0)), Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::TraitDef>(2) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("assoc_ty"))] }, input: None }]) }]
> TypeAlias { name: Name(Text("AssocTy")), visibility: RawVisibilityId("pub(self)"), bounds: [Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Tr"))] }, generic_args: [Some(GenericArgs { args: [Type(Tuple([]))], has_self_type: false, bindings: [] })] })], generic_params: GenericParamsId(4294967295), type_ref: None, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::TypeAliasDef>(8) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("assoc_const"))] }, input: None }]) }]
> Const { name: Some(Name(Text("CONST"))), visibility: RawVisibilityId("pub(self)"), type_ref: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("u8"))] }, generic_args: [None] }), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ConstDef>(9) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("assoc_method"))] }, input: None }]) }]
> Function { name: Name(Text("method")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: true, is_unsafe: false, params: [Reference(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Self"))] }, generic_args: [None] }), Shared)], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(10) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("assoc_dfl_method"))] }, input: None }]) }]
> Function { name: Name(Text("dfl_method")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: true, is_unsafe: false, params: [Reference(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Self"))] }, generic_args: [None] }), Mut)], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(11) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("struct0"))] }, input: None }]) }]
Struct { name: Name(Text("Struct0")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(1), fields: Unit, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(3), kind: Unit }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("struct1"))] }, input: None }]) }]
Struct { name: Name(Text("Struct1")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(2), fields: Tuple(IdRange::<ra_hir_def::item_tree::Field>(0..1)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(4), kind: Tuple }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("struct2"))] }, input: None }]) }]
Struct { name: Name(Text("Struct2")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(3), fields: Record(IdRange::<ra_hir_def::item_tree::Field>(1..2)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(5), kind: Record }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("en"))] }, input: None }]) }]
Enum { name: Name(Text("En")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), variants: IdRange::<ra_hir_def::item_tree::Variant>(0..1), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::EnumDef>(6) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("un"))] }, input: None }]) }]
Union { name: Name(Text("Un")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), fields: Record(IdRange::<ra_hir_def::item_tree::Field>(3..4)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UnionDef>(7) }
    "###);
}

#[test]
fn simple_inner_items() {
    let tree = print_item_tree(
        r"
        impl<T:A> D for Response<T> {
            fn foo() {
                end();
                fn end<W: Write>() {
                    let _x: T = loop {};
                }
            }
        }
    ",
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
Impl { generic_params: GenericParamsId(0), target_trait: Some(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("D"))] }, generic_args: [None] })), target_type: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Response"))] }, generic_args: [Some(GenericArgs { args: [Type(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("T"))] }, generic_args: [None] }))], has_self_type: false, bindings: [] })] }), is_negative: false, items: [Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ImplDef>(0) }
> Function { name: Name(Text("foo")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(1) }

inner items:

for AST FileAstId::<ra_syntax::ast::generated::nodes::ModuleItem>(2):
Function { name: Name(Text("end")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(1), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(2) }

    "###);
}

#[test]
fn extern_attrs() {
    let tree = print_item_tree(
        r#"
        #[block_attr]
        extern "C" {
            #[attr_a]
            fn a() {}
            #[attr_b]
            fn b() {}
        }
    "#,
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_a"))] }, input: None }, Attr { path: ModPath { kind: Plain, segments: [Name(Text("block_attr"))] }, input: None }]) }]
Function { name: Name(Text("a")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(1) }
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_b"))] }, input: None }, Attr { path: ModPath { kind: Plain, segments: [Name(Text("block_attr"))] }, input: None }]) }]
Function { name: Name(Text("b")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(2) }
    "###);
}

#[test]
fn trait_attrs() {
    let tree = print_item_tree(
        r#"
        #[trait_attr]
        trait Tr {
            #[attr_a]
            fn a() {}
            #[attr_b]
            fn b() {}
        }
    "#,
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("trait_attr"))] }, input: None }]) }]
Trait { name: Name(Text("Tr")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(0), auto: false, items: [Function(Idx::<Function>(0)), Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::TraitDef>(0) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_a"))] }, input: None }]) }]
> Function { name: Name(Text("a")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(1) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_b"))] }, input: None }]) }]
> Function { name: Name(Text("b")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(2) }
    "###);
}

#[test]
fn impl_attrs() {
    let tree = print_item_tree(
        r#"
        #[impl_attr]
        impl Ty {
            #[attr_a]
            fn a() {}
            #[attr_b]
            fn b() {}
        }
    "#,
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("impl_attr"))] }, input: None }]) }]
Impl { generic_params: GenericParamsId(4294967295), target_trait: None, target_type: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Ty"))] }, generic_args: [None] }), is_negative: false, items: [Function(Idx::<Function>(0)), Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ImplDef>(0) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_a"))] }, input: None }]) }]
> Function { name: Name(Text("a")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(1) }
> #[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr_b"))] }, input: None }]) }]
> Function { name: Name(Text("b")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(2) }
    "###);
}

#[test]
fn cursed_inner_items() {
    test_inner_items(
        r"
        struct S<T: Trait = [u8; { fn f() {} 0 }]>(T);

        enum En {
            Var1 {
                t: [(); { trait Inner {} 0 }],
            },

            Var2([u16; { enum Inner {} 0 }]),
        }

        type Ty = [En; { struct Inner; 0 }];

        impl En {
            fn assoc() {
                trait InnerTrait<T = [u8; { fn f() {} }]> {}
                struct InnerStruct<T = [u8; { fn f() {} }]> {}
                impl<T = [u8; { fn f() {} }]> InnerTrait for InnerStruct {}
            }
        }

        trait Tr<T = [u8; { fn f() {} }]> {
            type AssocTy = [u8; { fn f() {} }];

            const AssocConst: [u8; { fn f() {} }];
        }
    ",
    );
}

#[test]
fn inner_item_attrs() {
    let tree = print_item_tree(
        r"
        fn foo() {
            #[on_inner]
            fn inner() {}
        }
    ",
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
Function { name: Name(Text("foo")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(0) }

inner items:

for AST FileAstId::<ra_syntax::ast::generated::nodes::ModuleItem>(1):
#[Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("on_inner"))] }, input: None }]) }]
Function { name: Name(Text("inner")), visibility: RawVisibilityId("pub(self)"), generic_params: GenericParamsId(4294967295), has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(1) }

    "###);
}

#[test]
fn assoc_item_macros() {
    let tree = print_item_tree(
        r"
        impl S {
            items!();
        }
    ",
    );

    assert_snapshot!(tree, @r###"
inner attrs: Attrs { entries: None }

top-level items:
Impl { generic_params: GenericParamsId(4294967295), target_trait: None, target_type: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("S"))] }, generic_args: [None] }), is_negative: false, items: [MacroCall(Idx::<MacroCall>(0))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ImplDef>(0) }
> MacroCall { name: None, path: ModPath { kind: Plain, segments: [Name(Text("items"))] }, is_export: false, is_local_inner: false, is_builtin: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::MacroCall>(1) }
    "###);
}

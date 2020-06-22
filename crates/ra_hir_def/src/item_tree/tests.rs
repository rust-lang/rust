use super::{ItemTree, ModItem, ModKind};
use crate::{db::DefDatabase, test_db::TestDB};
use hir_expand::db::AstDatabase;
use insta::assert_snapshot;
use ra_db::fixture::WithFixture;
use ra_syntax::{ast, AstNode};
use rustc_hash::FxHashSet;
use std::sync::Arc;
use stdx::format_to;

fn test_inner_items(ra_fixture: &str) {
    let (db, file_id) = TestDB::with_single_file(ra_fixture);
    let tree = db.item_tree(file_id.into());
    let root = db.parse_or_expand(file_id.into()).unwrap();
    let ast_id_map = db.ast_id_map(file_id.into());

    // Traverse the item tree and collect all module/impl/trait-level items as AST nodes.
    let mut outer_items = FxHashSet::default();
    let mut worklist = tree.top_level_items().to_vec();
    while let Some(item) = worklist.pop() {
        let node: ast::ModuleItem = match item {
            ModItem::Import(it) => tree.source(&db, it).into(),
            ModItem::ExternCrate(it) => tree.source(&db, it).into(),
            ModItem::Function(it) => tree.source(&db, it).into(),
            ModItem::Struct(it) => tree.source(&db, it).into(),
            ModItem::Union(it) => tree.source(&db, it).into(),
            ModItem::Enum(it) => tree.source(&db, it).into(),
            ModItem::Const(it) => tree.source(&db, it).into(),
            ModItem::Static(it) => tree.source(&db, it).into(),
            ModItem::TypeAlias(it) => tree.source(&db, it).into(),
            ModItem::Mod(it) => {
                if let ModKind::Inline { items } = &tree[it].kind {
                    worklist.extend(items);
                }
                tree.source(&db, it).into()
            }
            ModItem::Trait(it) => {
                worklist.extend(tree[it].items.iter().map(|item| ModItem::from(*item)));
                tree.source(&db, it).into()
            }
            ModItem::Impl(it) => {
                worklist.extend(tree[it].items.iter().map(|item| ModItem::from(*item)));
                tree.source(&db, it).into()
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
        format_to!(out, "\ninner items:\n");
        for (ast_id, items) in &tree.inner_items {
            format_to!(out, "{:?}:\n", ast_id);
            for inner in items {
                format_to!(out, "- ");
                fmt_mod_item(&mut out, &tree, *inner);
                format_to!(out, "\n");
            }
        }
    }

    out
}

fn fmt_mod_item(out: &mut String, tree: &ItemTree, item: ModItem) {
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
        }
        ModItem::Impl(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::TypeAlias(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::Mod(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
        ModItem::MacroCall(it) => {
            format_to!(out, "{:?}", tree[it]);
        }
    }
}

#[test]
fn smoke() {
    assert_snapshot!(print_item_tree(r"
        #![attr]

        use {a, b::*};
        extern crate krate;

        trait Tr<U> {
            type AssocTy: Tr<()>;
            const CONST: u8;
            fn method(&self);
            fn dfl_method(&mut self) {}
        }

        struct Struct0<T = ()>;
        struct Struct1<T>(u8);
        struct Struct2<T> {
            fld: (T, ),
        }

        enum En {
            Variant {
                field: u8,
            },
        }

        union Un {
            fld: u16,
        }
    "), @r###"
inner attrs: Attrs { entries: Some([Attr { path: ModPath { kind: Plain, segments: [Name(Text("attr"))] }, input: None }]) }

top-level items:
Import { path: ModPath { kind: Plain, segments: [Name(Text("a"))] }, alias: None, visibility: Module(ModPath { kind: Super(0), segments: [] }), is_glob: false, is_prelude: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UseItem>(0) }
Import { path: ModPath { kind: Plain, segments: [Name(Text("b"))] }, alias: None, visibility: Module(ModPath { kind: Super(0), segments: [] }), is_glob: true, is_prelude: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UseItem>(0) }
ExternCrate { path: ModPath { kind: Plain, segments: [Name(Text("krate"))] }, alias: None, visibility: Module(ModPath { kind: Super(0), segments: [] }), is_macro_use: false, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ExternCrateItem>(1) }
Trait { name: Name(Text("Tr")), visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 2, data: [TypeParamData { name: Some(Name(Text("Self"))), default: None, provenance: TraitSelf }, TypeParamData { name: Some(Name(Text("U"))), default: None, provenance: TypeParamList }] }, where_predicates: [] }, auto: false, items: [TypeAlias(Idx::<TypeAlias>(0)), Const(Idx::<Const>(0)), Function(Idx::<Function>(0)), Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::TraitDef>(2) }
Struct { name: Name(Text("Struct0")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 1, data: [TypeParamData { name: Some(Name(Text("T"))), default: Some(Tuple([])), provenance: TypeParamList }] }, where_predicates: [] }, fields: Unit, ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(3), kind: Unit }
Struct { name: Name(Text("Struct1")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 1, data: [TypeParamData { name: Some(Name(Text("T"))), default: None, provenance: TypeParamList }] }, where_predicates: [] }, fields: Tuple(Idx::<Field>(0)..Idx::<Field>(1)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(4), kind: Tuple }
Struct { name: Name(Text("Struct2")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 1, data: [TypeParamData { name: Some(Name(Text("T"))), default: None, provenance: TypeParamList }] }, where_predicates: [] }, fields: Record(Idx::<Field>(1)..Idx::<Field>(2)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::StructDef>(5), kind: Record }
Enum { name: Name(Text("En")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 0, data: [] }, where_predicates: [] }, variants: Idx::<Variant>(0)..Idx::<Variant>(1), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::EnumDef>(6) }
Union { name: Name(Text("Un")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 0, data: [] }, where_predicates: [] }, fields: Record(Idx::<Field>(3)..Idx::<Field>(4)), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::UnionDef>(7) }
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
Impl { generic_params: GenericParams { types: Arena { len: 1, data: [TypeParamData { name: Some(Name(Text("T"))), default: None, provenance: TypeParamList }] }, where_predicates: [WherePredicate { target: TypeRef(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("T"))] }, generic_args: [None] })), bound: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("A"))] }, generic_args: [None] }) }] }, target_trait: Some(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("D"))] }, generic_args: [None] })), target_type: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Response"))] }, generic_args: [Some(GenericArgs { args: [Type(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("T"))] }, generic_args: [None] }))], has_self_type: false, bindings: [] })] }), is_negative: false, items: [Function(Idx::<Function>(1))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ImplDef>(0) }

inner items:
FileAstId::<ra_syntax::ast::generated::nodes::ModuleItem>(2):
- Function { name: Name(Text("end")), attrs: Attrs { entries: None }, visibility: Module(ModPath { kind: Super(0), segments: [] }), generic_params: GenericParams { types: Arena { len: 1, data: [TypeParamData { name: Some(Name(Text("W"))), default: None, provenance: TypeParamList }] }, where_predicates: [WherePredicate { target: TypeRef(Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("W"))] }, generic_args: [None] })), bound: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("Write"))] }, generic_args: [None] }) }] }, has_self_param: false, is_unsafe: false, params: [], ret_type: Tuple([]), ast_id: FileAstId::<ra_syntax::ast::generated::nodes::FnDef>(2) }
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
                trait InnerTrait {}
                struct InnerStruct {}
                impl InnerTrait for InnerStruct {}
            }
        }
    ",
    );
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
Impl { generic_params: GenericParams { types: Arena { len: 0, data: [] }, where_predicates: [] }, target_trait: None, target_type: Path(Path { type_anchor: None, mod_path: ModPath { kind: Plain, segments: [Name(Text("S"))] }, generic_args: [None] }), is_negative: false, items: [MacroCall(Idx::<MacroCall>(0))], ast_id: FileAstId::<ra_syntax::ast::generated::nodes::ImplDef>(0) }
    "###);
}

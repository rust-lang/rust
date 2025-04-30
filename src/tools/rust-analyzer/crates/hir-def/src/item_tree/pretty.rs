//! `ItemTree` debug printer.

use std::fmt::{self, Write};

use la_arena::{Idx, RawIdx};
use span::{Edition, ErasedFileAstId};

use crate::{
    item_tree::{
        AttrOwner, Const, DefDatabase, Enum, ExternBlock, ExternCrate, Field, FieldParent,
        FieldsShape, FileItemTreeId, Function, Impl, ItemTree, Macro2, MacroCall, MacroRules, Mod,
        ModItem, ModKind, RawAttrs, RawVisibilityId, Static, Struct, Trait, TraitAlias, TypeAlias,
        Union, Use, UseTree, UseTreeKind, Variant,
    },
    visibility::RawVisibility,
};

pub(super) fn print_item_tree(db: &dyn DefDatabase, tree: &ItemTree, edition: Edition) -> String {
    let mut p =
        Printer { db, tree, buf: String::new(), indent_level: 0, needs_indent: true, edition };

    if let Some(attrs) = tree.attrs.get(&AttrOwner::TopLevel) {
        p.print_attrs(attrs, true, "\n");
    }
    p.blank();

    for item in tree.top_level_items() {
        p.print_mod_item(*item);
    }

    let mut s = p.buf.trim_end_matches('\n').to_owned();
    s.push('\n');
    s
}

macro_rules! w {
    ($dst:expr, $($arg:tt)*) => {
        { let _ = write!($dst, $($arg)*); }
    };
}

macro_rules! wln {
    ($dst:expr) => {
        { let _ = writeln!($dst); }
    };
    ($dst:expr, $($arg:tt)*) => {
        { let _ = writeln!($dst, $($arg)*); }
    };
}

struct Printer<'a> {
    db: &'a dyn DefDatabase,
    tree: &'a ItemTree,
    buf: String,
    indent_level: usize,
    needs_indent: bool,
    edition: Edition,
}

impl Printer<'_> {
    fn indented(&mut self, f: impl FnOnce(&mut Self)) {
        self.indent_level += 1;
        wln!(self);
        f(self);
        self.indent_level -= 1;
        self.buf = self.buf.trim_end_matches('\n').to_owned();
    }

    /// Ensures that a blank line is output before the next text.
    fn blank(&mut self) {
        let mut iter = self.buf.chars().rev().fuse();
        match (iter.next(), iter.next()) {
            (Some('\n'), Some('\n') | None) | (None, None) => {}
            (Some('\n'), Some(_)) => {
                self.buf.push('\n');
            }
            (Some(_), _) => {
                self.buf.push('\n');
                self.buf.push('\n');
            }
            (None, Some(_)) => unreachable!(),
        }
    }

    fn whitespace(&mut self) {
        match self.buf.chars().next_back() {
            None | Some('\n' | ' ') => {}
            _ => self.buf.push(' '),
        }
    }

    fn print_attrs(&mut self, attrs: &RawAttrs, inner: bool, separated_by: &str) {
        let inner = if inner { "!" } else { "" };
        for attr in &**attrs {
            w!(
                self,
                "#{}[{}{}]{}",
                inner,
                attr.path.display(self.db, self.edition),
                attr.input.as_ref().map(|it| it.to_string()).unwrap_or_default(),
                separated_by,
            );
        }
    }

    fn print_attrs_of(&mut self, of: impl Into<AttrOwner>, separated_by: &str) {
        if let Some(attrs) = self.tree.attrs.get(&of.into()) {
            self.print_attrs(attrs, false, separated_by);
        }
    }

    fn print_visibility(&mut self, vis: RawVisibilityId) {
        match &self.tree[vis] {
            RawVisibility::Module(path, _expl) => {
                w!(self, "pub({}) ", path.display(self.db, self.edition))
            }
            RawVisibility::Public => w!(self, "pub "),
        };
    }

    fn print_fields(&mut self, parent: FieldParent, kind: FieldsShape, fields: &[Field]) {
        let edition = self.edition;
        match kind {
            FieldsShape::Record => {
                self.whitespace();
                w!(self, "{{");
                self.indented(|this| {
                    for (idx, Field { name, visibility, is_unsafe }) in fields.iter().enumerate() {
                        this.print_attrs_of(
                            AttrOwner::Field(parent, Idx::from_raw(RawIdx::from(idx as u32))),
                            "\n",
                        );
                        this.print_visibility(*visibility);
                        if *is_unsafe {
                            w!(this, "unsafe ");
                        }

                        wln!(this, "{},", name.display(self.db, edition));
                    }
                });
                w!(self, "}}");
            }
            FieldsShape::Tuple => {
                w!(self, "(");
                self.indented(|this| {
                    for (idx, Field { name, visibility, is_unsafe }) in fields.iter().enumerate() {
                        this.print_attrs_of(
                            AttrOwner::Field(parent, Idx::from_raw(RawIdx::from(idx as u32))),
                            "\n",
                        );
                        this.print_visibility(*visibility);
                        if *is_unsafe {
                            w!(this, "unsafe ");
                        }
                        wln!(this, "{},", name.display(self.db, edition));
                    }
                });
                w!(self, ")");
            }
            FieldsShape::Unit => {}
        }
    }

    fn print_use_tree(&mut self, use_tree: &UseTree) {
        match &use_tree.kind {
            UseTreeKind::Single { path, alias } => {
                w!(self, "{}", path.display(self.db, self.edition));
                if let Some(alias) = alias {
                    w!(self, " as {}", alias.display(self.edition));
                }
            }
            UseTreeKind::Glob { path } => {
                if let Some(path) = path {
                    w!(self, "{}::", path.display(self.db, self.edition));
                }
                w!(self, "*");
            }
            UseTreeKind::Prefixed { prefix, list } => {
                if let Some(prefix) = prefix {
                    w!(self, "{}::", prefix.display(self.db, self.edition));
                }
                w!(self, "{{");
                for (i, tree) in list.iter().enumerate() {
                    if i != 0 {
                        w!(self, ", ");
                    }
                    self.print_use_tree(tree);
                }
                w!(self, "}}");
            }
        }
    }

    fn print_mod_item(&mut self, item: ModItem) {
        self.print_attrs_of(item, "\n");

        match item {
            ModItem::Use(it) => {
                let Use { visibility, use_tree, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "use ");
                self.print_use_tree(use_tree);
                wln!(self, ";");
            }
            ModItem::ExternCrate(it) => {
                let ExternCrate { name, alias, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "extern crate {}", name.display(self.db, self.edition));
                if let Some(alias) = alias {
                    w!(self, " as {}", alias.display(self.edition));
                }
                wln!(self, ";");
            }
            ModItem::ExternBlock(it) => {
                let ExternBlock { abi, ast_id, children } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                w!(self, "extern ");
                if let Some(abi) = abi {
                    w!(self, "\"{}\" ", abi);
                }
                w!(self, "{{");
                self.indented(|this| {
                    for child in &**children {
                        this.print_mod_item(*child);
                    }
                });
                wln!(self, "}}");
            }
            ModItem::Function(it) => {
                let Function { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "fn {};", name.display(self.db, self.edition));
            }
            ModItem::Struct(it) => {
                let Struct { visibility, name, fields, shape: kind, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "struct {}", name.display(self.db, self.edition));
                self.print_fields(FieldParent::Struct(it), *kind, fields);
                if matches!(kind, FieldsShape::Record) {
                    wln!(self);
                } else {
                    wln!(self, ";");
                }
            }
            ModItem::Union(it) => {
                let Union { name, visibility, fields, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "union {}", name.display(self.db, self.edition));
                self.print_fields(FieldParent::Union(it), FieldsShape::Record, fields);
                wln!(self);
            }
            ModItem::Enum(it) => {
                let Enum { name, visibility, variants, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "enum {}", name.display(self.db, self.edition));
                let edition = self.edition;
                self.indented(|this| {
                    for variant in FileItemTreeId::range_iter(variants.clone()) {
                        let Variant { name, fields, shape: kind, ast_id } = &this.tree[variant];
                        this.print_ast_id(ast_id.erase());
                        this.print_attrs_of(variant, "\n");
                        w!(this, "{}", name.display(self.db, edition));
                        this.print_fields(FieldParent::EnumVariant(variant), *kind, fields);
                        wln!(this, ",");
                    }
                });
                wln!(self, "}}");
            }
            ModItem::Const(it) => {
                let Const { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "const ");
                match name {
                    Some(name) => w!(self, "{}", name.display(self.db, self.edition)),
                    None => w!(self, "_"),
                }
                wln!(self, " = _;");
            }
            ModItem::Static(it) => {
                let Static { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "static ");
                w!(self, "{}", name.display(self.db, self.edition));
                w!(self, " = _;");
                wln!(self);
            }
            ModItem::Trait(it) => {
                let Trait { name, visibility, items, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "trait {} {{", name.display(self.db, self.edition));
                self.indented(|this| {
                    for item in &**items {
                        this.print_mod_item((*item).into());
                    }
                });
                wln!(self, "}}");
            }
            ModItem::TraitAlias(it) => {
                let TraitAlias { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "trait {} = ..;", name.display(self.db, self.edition));
            }
            ModItem::Impl(it) => {
                let Impl { items, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                w!(self, "impl {{");
                self.indented(|this| {
                    for item in &**items {
                        this.print_mod_item((*item).into());
                    }
                });
                wln!(self, "}}");
            }
            ModItem::TypeAlias(it) => {
                let TypeAlias { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "type {}", name.display(self.db, self.edition));
                w!(self, ";");
                wln!(self);
            }
            ModItem::Mod(it) => {
                let Mod { name, visibility, kind, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "mod {}", name.display(self.db, self.edition));
                match kind {
                    ModKind::Inline { items } => {
                        w!(self, " {{");
                        self.indented(|this| {
                            for item in &**items {
                                this.print_mod_item(*item);
                            }
                        });
                        wln!(self, "}}");
                    }
                    ModKind::Outline => {
                        wln!(self, ";");
                    }
                }
            }
            ModItem::MacroCall(it) => {
                let MacroCall { path, ast_id, expand_to, ctxt } = &self.tree[it];
                let _ = writeln!(
                    self,
                    "// AstId: {:?}, SyntaxContextId: {}, ExpandTo: {:?}",
                    ast_id.erase().into_raw(),
                    ctxt,
                    expand_to
                );
                wln!(self, "{}!(...);", path.display(self.db, self.edition));
            }
            ModItem::MacroRules(it) => {
                let MacroRules { name, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                wln!(self, "macro_rules! {} {{ ... }}", name.display(self.db, self.edition));
            }
            ModItem::Macro2(it) => {
                let Macro2 { name, visibility, ast_id } = &self.tree[it];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "macro {} {{ ... }}", name.display(self.db, self.edition));
            }
        }

        self.blank();
    }

    fn print_ast_id(&mut self, ast_id: ErasedFileAstId) {
        wln!(self, "// AstId: {:?}", ast_id.into_raw());
    }
}

impl Write for Printer<'_> {
    fn write_str(&mut self, s: &str) -> fmt::Result {
        for line in s.split_inclusive('\n') {
            if self.needs_indent {
                match self.buf.chars().last() {
                    Some('\n') | None => {}
                    _ => self.buf.push('\n'),
                }
                self.buf.push_str(&"    ".repeat(self.indent_level));
                self.needs_indent = false;
            }

            self.buf.push_str(line);
            self.needs_indent = line.ends_with('\n');
        }

        Ok(())
    }
}

//! `ItemTree` debug printer.

use std::fmt::{self, Write};

use span::{Edition, ErasedFileAstId};

use crate::{
    item_tree::{
        Const, DefDatabase, Enum, ExternBlock, ExternCrate, FieldsShape, Function, Impl, ItemTree,
        Macro2, MacroCall, MacroRules, Mod, ModItemId, ModKind, RawAttrs, RawVisibilityId, Static,
        Struct, Trait, TraitAlias, TypeAlias, Union, Use, UseTree, UseTreeKind,
    },
    visibility::RawVisibility,
};

pub(super) fn print_item_tree(db: &dyn DefDatabase, tree: &ItemTree, edition: Edition) -> String {
    let mut p =
        Printer { db, tree, buf: String::new(), indent_level: 0, needs_indent: true, edition };

    p.print_attrs(&tree.top_attrs, true, "\n");
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

    fn print_attrs_of(&mut self, of: ModItemId, separated_by: &str) {
        if let Some(attrs) = self.tree.attrs.get(&of.ast_id()) {
            self.print_attrs(attrs, false, separated_by);
        }
    }

    fn print_visibility(&mut self, vis: RawVisibilityId) {
        match &self.tree[vis] {
            RawVisibility::Module(path, _expl) => {
                w!(self, "pub(in {}) ", path.display(self.db, self.edition))
            }
            RawVisibility::Public => w!(self, "pub "),
            RawVisibility::PubCrate => w!(self, "pub(crate) "),
            RawVisibility::PubSelf(_) => w!(self, "pub(self) "),
        };
    }

    fn print_fields(&mut self, kind: FieldsShape) {
        match kind {
            FieldsShape::Record => {
                self.whitespace();
                w!(self, "{{ ... }}");
            }
            FieldsShape::Tuple => {
                w!(self, "(...)");
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

    fn print_mod_item(&mut self, item: ModItemId) {
        self.print_attrs_of(item, "\n");

        match item {
            ModItemId::Use(ast_id) => {
                let Use { visibility, use_tree } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "use ");
                self.print_use_tree(use_tree);
                wln!(self, ";");
            }
            ModItemId::ExternCrate(ast_id) => {
                let ExternCrate { name, alias, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "extern crate {}", name.display(self.db, self.edition));
                if let Some(alias) = alias {
                    w!(self, " as {}", alias.display(self.edition));
                }
                wln!(self, ";");
            }
            ModItemId::ExternBlock(ast_id) => {
                let ExternBlock { children } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                w!(self, "extern {{");
                self.indented(|this| {
                    for child in &**children {
                        this.print_mod_item(*child);
                    }
                });
                wln!(self, "}}");
            }
            ModItemId::Function(ast_id) => {
                let Function { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "fn {};", name.display(self.db, self.edition));
            }
            ModItemId::Struct(ast_id) => {
                let Struct { visibility, name, shape: kind } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "struct {}", name.display(self.db, self.edition));
                self.print_fields(*kind);
                if matches!(kind, FieldsShape::Record) {
                    wln!(self);
                } else {
                    wln!(self, ";");
                }
            }
            ModItemId::Union(ast_id) => {
                let Union { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "union {}", name.display(self.db, self.edition));
                self.print_fields(FieldsShape::Record);
                wln!(self);
            }
            ModItemId::Enum(ast_id) => {
                let Enum { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "enum {} {{ ... }}", name.display(self.db, self.edition));
            }
            ModItemId::Const(ast_id) => {
                let Const { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "const ");
                match name {
                    Some(name) => w!(self, "{}", name.display(self.db, self.edition)),
                    None => w!(self, "_"),
                }
                wln!(self, " = _;");
            }
            ModItemId::Static(ast_id) => {
                let Static { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "static ");
                w!(self, "{}", name.display(self.db, self.edition));
                w!(self, " = _;");
                wln!(self);
            }
            ModItemId::Trait(ast_id) => {
                let Trait { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "trait {} {{ ... }}", name.display(self.db, self.edition));
            }
            ModItemId::TraitAlias(ast_id) => {
                let TraitAlias { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "trait {} = ..;", name.display(self.db, self.edition));
            }
            ModItemId::Impl(ast_id) => {
                let Impl {} = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                w!(self, "impl {{ ... }}");
            }
            ModItemId::TypeAlias(ast_id) => {
                let TypeAlias { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                w!(self, "type {}", name.display(self.db, self.edition));
                w!(self, ";");
                wln!(self);
            }
            ModItemId::Mod(ast_id) => {
                let Mod { name, visibility, kind } = &self.tree[ast_id];
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
            ModItemId::MacroCall(ast_id) => {
                let MacroCall { path, expand_to, ctxt } = &self.tree[ast_id];
                let _ = writeln!(
                    self,
                    "// AstId: {:#?}, SyntaxContextId: {}, ExpandTo: {:?}",
                    ast_id.erase(),
                    ctxt,
                    expand_to
                );
                wln!(self, "{}!(...);", path.display(self.db, self.edition));
            }
            ModItemId::MacroRules(ast_id) => {
                let MacroRules { name } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                wln!(self, "macro_rules! {} {{ ... }}", name.display(self.db, self.edition));
            }
            ModItemId::Macro2(ast_id) => {
                let Macro2 { name, visibility } = &self.tree[ast_id];
                self.print_ast_id(ast_id.erase());
                self.print_visibility(*visibility);
                wln!(self, "macro {} {{ ... }}", name.display(self.db, self.edition));
            }
        }

        self.blank();
    }

    fn print_ast_id(&mut self, ast_id: ErasedFileAstId) {
        wln!(self, "// AstId: {ast_id:#?}");
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

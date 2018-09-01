use libsyntax2::{
    AstNode, SyntaxNode, SmolStr, ast
};

pub struct ModuleScope {
    entries: Vec<Entry>,
}

pub struct Entry {
    node: SyntaxNode,
    kind: EntryKind,
}

enum EntryKind {
    Item, Import,
}

impl ModuleScope {
    pub fn new(m: ast::Root) -> ModuleScope {
        let mut entries = Vec::new();
        for item in m.items() {
            let entry = match item {
                ast::ModuleItem::StructDef(item) => Entry::new(item),
                ast::ModuleItem::EnumDef(item) => Entry::new(item),
                ast::ModuleItem::FnDef(item) => Entry::new(item),
                ast::ModuleItem::ConstDef(item) => Entry::new(item),
                ast::ModuleItem::StaticDef(item) => Entry::new(item),
                ast::ModuleItem::TraitDef(item) => Entry::new(item),
                ast::ModuleItem::TypeDef(item) => Entry::new(item),
                ast::ModuleItem::Module(item) => Entry::new(item),
                ast::ModuleItem::UseItem(item) => {
                    if let Some(tree) = item.use_tree() {
                        collect_imports(tree, &mut entries);
                    }
                    continue;
                },
                ast::ModuleItem::ExternCrateItem(_) |
                ast::ModuleItem::ImplItem(_) => continue,
            };
            entries.extend(entry)
        }

        ModuleScope { entries }
    }

    pub fn entries(&self) -> &[Entry] {
        self.entries.as_slice()
    }
}

impl Entry {
    fn new<'a>(item: impl ast::NameOwner<'a>) -> Option<Entry> {
        let name = item.name()?;
        Some(Entry { node: name.syntax().owned(), kind: EntryKind::Item })
    }
    fn new_import(path: ast::Path) -> Option<Entry> {
        let name_ref = path.segment()?.name_ref()?;
        Some(Entry { node: name_ref.syntax().owned(), kind: EntryKind::Import })
    }
    pub fn name(&self) -> SmolStr {
        match self.kind {
            EntryKind::Item =>
                ast::Name::cast(self.node.borrowed()).unwrap()
                    .text(),
            EntryKind::Import =>
                ast::NameRef::cast(self.node.borrowed()).unwrap()
                    .text(),
        }
    }
}

fn collect_imports(tree: ast::UseTree, acc: &mut Vec<Entry>) {
    if let Some(use_tree_list) = tree.use_tree_list() {
        return use_tree_list.use_trees().for_each(|it| collect_imports(it, acc));
    }
    if let Some(path) = tree.path() {
        acc.extend(Entry::new_import(path));
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use libsyntax2::File;

    fn do_check(code: &str, expected: &[&str]) {
        let file = File::parse(&code);
        let scope = ModuleScope::new(file.ast());
        let actual = scope.entries
            .iter()
            .map(|it| it.name())
            .collect::<Vec<_>>();
        assert_eq!(expected, actual.as_slice());
    }

    #[test]
    fn test_module_scope() {
        do_check("
            struct Foo;
            enum Bar {}
            mod baz {}
            fn quux() {}
            use x::{
                y::z,
                t,
            };
            type T = ();
        ", &["Foo", "Bar", "baz", "quux", "z", "t", "T"])
    }
}

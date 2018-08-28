use libsyntax2::{
    AstNode, SyntaxNode, SmolStr, ast
};

pub struct ModuleScope {
    entries: Vec<Entry>,
}

impl ModuleScope {
    pub fn new(m: ast::Root) -> ModuleScope {
        let entries = m.items().filter_map(|item| {
            match item {
                ast::ModuleItem::StructDef(item) => Entry::new(item),
                ast::ModuleItem::EnumDef(item) => Entry::new(item),
                ast::ModuleItem::FnDef(item) => Entry::new(item),
                ast::ModuleItem::TraitDef(item) => Entry::new(item),
                ast::ModuleItem::ExternCrateItem(_) |
                ast::ModuleItem::ImplItem(_) |
                ast::ModuleItem::UseItem(_) => None
            }
        }).collect();

        ModuleScope { entries }
    }

    pub fn entries(&self) -> &[Entry] {
        self.entries.as_slice()
    }
}

pub struct Entry {
    name: SyntaxNode,
}

impl Entry {
    fn new<'a>(item: impl ast::NameOwner<'a>) -> Option<Entry> {
        let name = item.name()?;
        Some(Entry { name: name.syntax().owned() })
    }
    pub fn name(&self) -> SmolStr {
        self.ast().text()
    }
    fn ast(&self) -> ast::Name {
        ast::Name::cast(self.name.borrowed()).unwrap()
    }
}


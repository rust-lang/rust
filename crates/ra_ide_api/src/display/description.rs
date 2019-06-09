use ra_syntax::{
    ast::{self, NameOwner, VisibilityOwner, TypeAscriptionOwner, AstNode},
};

pub(crate) trait Description {
    fn description(&self) -> Option<String>;
}

impl Description for ast::FnDef {
    fn description(&self) -> Option<String> {
        Some(crate::display::function_label(self))
    }
}

impl Description for ast::StructDef {
    fn description(&self) -> Option<String> {
        description_from_node(self, "struct ")
    }
}

impl Description for ast::EnumDef {
    fn description(&self) -> Option<String> {
        description_from_node(self, "enum ")
    }
}

impl Description for ast::TraitDef {
    fn description(&self) -> Option<String> {
        description_from_node(self, "trait ")
    }
}

impl Description for ast::Module {
    fn description(&self) -> Option<String> {
        description_from_node(self, "mod ")
    }
}

impl Description for ast::TypeAliasDef {
    fn description(&self) -> Option<String> {
        description_from_node(self, "type ")
    }
}

impl Description for ast::ConstDef {
    fn description(&self) -> Option<String> {
        description_from_ascribed_node(self, "const ")
    }
}

impl Description for ast::StaticDef {
    fn description(&self) -> Option<String> {
        description_from_ascribed_node(self, "static ")
    }
}

impl Description for ast::NamedFieldDef {
    fn description(&self) -> Option<String> {
        description_from_ascribed_node(self, "")
    }
}

impl Description for ast::EnumVariant {
    fn description(&self) -> Option<String> {
        Some(self.name()?.text().to_string())
    }
}

fn description_from_ascribed_node<T>(node: &T, prefix: &str) -> Option<String>
where
    T: NameOwner + VisibilityOwner + TypeAscriptionOwner,
{
    let mut string = description_from_node(node, prefix)?;

    if let Some(type_ref) = node.ascribed_type() {
        string.push_str(": ");
        type_ref.syntax().text().push_to(&mut string);
    }

    Some(string)
}

fn description_from_node<T>(node: &T, label: &str) -> Option<String>
where
    T: NameOwner + VisibilityOwner,
{
    let mut string =
        node.visibility().map(|v| format!("{} ", v.syntax().text())).unwrap_or_default();
    string.push_str(label);
    string.push_str(node.name()?.text().as_str());
    Some(string)
}

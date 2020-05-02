//! FIXME: write short doc here

use ra_syntax::ast::{self, AstNode, NameOwner, TypeAscriptionOwner, VisibilityOwner};
use stdx::format_to;

pub(crate) trait ShortLabel {
    fn short_label(&self) -> Option<String>;
}

impl ShortLabel for ast::FnDef {
    fn short_label(&self) -> Option<String> {
        Some(crate::display::function_label(self))
    }
}

impl ShortLabel for ast::StructDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "struct ")
    }
}

impl ShortLabel for ast::UnionDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "union ")
    }
}

impl ShortLabel for ast::EnumDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "enum ")
    }
}

impl ShortLabel for ast::TraitDef {
    fn short_label(&self) -> Option<String> {
        if self.unsafe_token().is_some() {
            short_label_from_node(self, "unsafe trait ")
        } else {
            short_label_from_node(self, "trait ")
        }
    }
}

impl ShortLabel for ast::Module {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "mod ")
    }
}

impl ShortLabel for ast::TypeAliasDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "type ")
    }
}

impl ShortLabel for ast::ConstDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_ascribed_node(self, "const ")
    }
}

impl ShortLabel for ast::StaticDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_ascribed_node(self, "static ")
    }
}

impl ShortLabel for ast::RecordFieldDef {
    fn short_label(&self) -> Option<String> {
        short_label_from_ascribed_node(self, "")
    }
}

impl ShortLabel for ast::EnumVariant {
    fn short_label(&self) -> Option<String> {
        Some(self.name()?.text().to_string())
    }
}

fn short_label_from_ascribed_node<T>(node: &T, prefix: &str) -> Option<String>
where
    T: NameOwner + VisibilityOwner + TypeAscriptionOwner,
{
    let mut buf = short_label_from_node(node, prefix)?;

    if let Some(type_ref) = node.ascribed_type() {
        format_to!(buf, ": {}", type_ref.syntax());
    }

    Some(buf)
}

fn short_label_from_node<T>(node: &T, label: &str) -> Option<String>
where
    T: NameOwner + VisibilityOwner,
{
    let mut buf = node.visibility().map(|v| format!("{} ", v.syntax())).unwrap_or_default();
    buf.push_str(label);
    buf.push_str(node.name()?.text().as_str());
    Some(buf)
}

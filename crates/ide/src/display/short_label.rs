//! FIXME: write short doc here

use stdx::format_to;
use syntax::ast::{self, AstNode, NameOwner, VisibilityOwner};

pub(crate) trait ShortLabel {
    fn short_label(&self) -> Option<String>;
}

impl ShortLabel for ast::Fn {
    fn short_label(&self) -> Option<String> {
        Some(crate::display::function_declaration(self))
    }
}

impl ShortLabel for ast::Struct {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "struct ")
    }
}

impl ShortLabel for ast::Union {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "union ")
    }
}

impl ShortLabel for ast::Enum {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "enum ")
    }
}

impl ShortLabel for ast::Trait {
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

impl ShortLabel for ast::SourceFile {
    fn short_label(&self) -> Option<String> {
        None
    }
}

impl ShortLabel for ast::BlockExpr {
    fn short_label(&self) -> Option<String> {
        None
    }
}

impl ShortLabel for ast::TypeAlias {
    fn short_label(&self) -> Option<String> {
        short_label_from_node(self, "type ")
    }
}

impl ShortLabel for ast::Const {
    fn short_label(&self) -> Option<String> {
        let mut new_buf = short_label_from_ty(self, self.ty(), "const ")?;
        if let Some(expr) = self.body() {
            format_to!(new_buf, " = {}", expr.syntax());
        }
        Some(new_buf)
    }
}

impl ShortLabel for ast::Static {
    fn short_label(&self) -> Option<String> {
        short_label_from_ty(self, self.ty(), "static ")
    }
}

impl ShortLabel for ast::RecordField {
    fn short_label(&self) -> Option<String> {
        short_label_from_ty(self, self.ty(), "")
    }
}

impl ShortLabel for ast::Variant {
    fn short_label(&self) -> Option<String> {
        Some(self.name()?.text().to_string())
    }
}

impl ShortLabel for ast::ConstParam {
    fn short_label(&self) -> Option<String> {
        let mut buf = "const ".to_owned();
        buf.push_str(self.name()?.text());
        if let Some(type_ref) = self.ty() {
            format_to!(buf, ": {}", type_ref.syntax());
        }
        Some(buf)
    }
}

fn short_label_from_ty<T>(node: &T, ty: Option<ast::Type>, prefix: &str) -> Option<String>
where
    T: NameOwner + VisibilityOwner,
{
    let mut buf = short_label_from_node(node, prefix)?;

    if let Some(type_ref) = ty {
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
    buf.push_str(node.name()?.text());
    Some(buf)
}

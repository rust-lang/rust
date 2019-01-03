use ra_syntax::{
    AstNode,
    ast::{self, VisibilityOwner, NameOwner},
    SyntaxKind::{VISIBILITY, FN_KW, MOD_KW, STRUCT_KW, ENUM_KW, TRAIT_KW, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF, IDENT},
};

use crate::assists::{AssistCtx, Assist};

pub fn change_visibility(ctx: AssistCtx) -> Option<Assist> {
    let offset = if let Some(keyword) = ctx.leaf_at_offset().find(|leaf| match leaf.kind() {
        FN_KW | MOD_KW | STRUCT_KW | ENUM_KW | TRAIT_KW => true,
        _ => false,
    }) {
        let parent = keyword.parent()?;
        let def_kws = vec![FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF];
        // Parent is not a definition, can't add visibility
        if !def_kws.iter().any(|&def_kw| def_kw == parent.kind()) {
            return None;
        }
        // Already have visibility, do nothing
        if parent.children().any(|child| child.kind() == VISIBILITY) {
            return None;
        }
        parent.range().start()
    } else {
        let ident = ctx.leaf_at_offset().find(|leaf| leaf.kind() == IDENT)?;
        let field = ident.ancestors().find_map(ast::NamedFieldDef::cast)?;
        if field.name()?.syntax().range() != ident.range() && field.visibility().is_some() {
            return None;
        }
        field.syntax().range().start()
    };

    ctx.build("make pub(crate)", |edit| {
        edit.insert(offset, "pub(crate) ");
        edit.set_cursor(offset);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assists::check_assist;

    #[test]
    fn change_visibility_adds_pub_crate_to_items() {
        check_assist(
            change_visibility,
            "<|>fn foo() {}",
            "<|>pub(crate) fn foo() {}",
        );
        check_assist(
            change_visibility,
            "f<|>n foo() {}",
            "<|>pub(crate) fn foo() {}",
        );
        check_assist(
            change_visibility,
            "<|>struct Foo {}",
            "<|>pub(crate) struct Foo {}",
        );
        check_assist(
            change_visibility,
            "<|>mod foo {}",
            "<|>pub(crate) mod foo {}",
        );
        check_assist(
            change_visibility,
            "<|>trait Foo {}",
            "<|>pub(crate) trait Foo {}",
        );
        check_assist(change_visibility, "m<|>od {}", "<|>pub(crate) mod {}");
        check_assist(
            change_visibility,
            "unsafe f<|>n foo() {}",
            "<|>pub(crate) unsafe fn foo() {}",
        );
    }

    #[test]
    fn change_visibility_works_with_struct_fields() {
        check_assist(
            change_visibility,
            "struct S { <|>field: u32 }",
            "struct S { <|>pub(crate) field: u32 }",
        )
    }
}

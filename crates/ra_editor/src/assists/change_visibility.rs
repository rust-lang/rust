use ra_syntax::{
    SyntaxKind::{VISIBILITY, FN_KW, MOD_KW, STRUCT_KW, ENUM_KW, TRAIT_KW, FN_DEF, MODULE, STRUCT_DEF, ENUM_DEF, TRAIT_DEF},
};

use crate::assists::{AssistCtx, Assist};

pub fn change_visibility(ctx: AssistCtx) -> Option<Assist> {
    let keyword = ctx.leaf_at_offset().find(|leaf| match leaf.kind() {
        FN_KW | MOD_KW | STRUCT_KW | ENUM_KW | TRAIT_KW => true,
        _ => false,
    })?;
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

    let node_start = parent.range().start();
    ctx.build("make pub crate", |edit| {
        edit.insert(node_start, "pub(crate) ");
        edit.set_cursor(node_start);
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assists::check_assist;

    #[test]
    fn test_change_visibility() {
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
}

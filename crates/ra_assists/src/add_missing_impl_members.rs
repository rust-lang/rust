use crate::assist_ctx::{Assist, AssistCtx};
use hir::db::HirDatabase;

pub(crate) fn add_missing_impl_members(mut ctx: AssistCtx<impl HirDatabase>) -> Option<Assist> {
    unimplemented!()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::helpers::{ check_assist };

    #[test]
    fn test_add_missing_impl_members() {
        check_assist(
            add_missing_impl_members,
            "
trait Foo {
    fn foo(&self);
}

struct S;

impl Foo for S {
    <|>
}",
            "
trait Foo {
    fn foo(&self);
}

struct S;

impl Foo for S {
    fn foo(&self) {
        <|>
    }
}",
        );
    }
}

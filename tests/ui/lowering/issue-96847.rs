//@ check-fail

// Test that this doesn't abort during AST lowering. In #96847 it did abort
// because the attribute was being lowered twice.

#![feature(stmt_expr_attributes)]
#![feature(lang_items)]

fn main() {
    for _ in [1,2,3] {
        #![lang="foo"]
        //~^ ERROR definition of an unknown lang item: `foo` [E0522]
        println!("foo");
    }
}

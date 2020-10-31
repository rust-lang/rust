// compile-flags: -Zast-json
//
// Regression test for issue #78510
// Tests that we don't ICE when we have tokens for an associated item

struct S;

impl S {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to structs, enums and unions
    fn f() {}
}

trait Bar {
    #[derive(Debug)] //~ ERROR `derive` may only be applied to structs, enums and unions
    fn foo() {}
}

fn main() {}

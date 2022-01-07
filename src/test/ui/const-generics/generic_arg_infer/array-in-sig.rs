// To avoid having to `or` gate `_` as an expr.
#![feature(generic_arg_infer)]

fn foo() -> [u8; _] {
    //~^ ERROR the const placeholder `_` is not allowed within types on item signatures for generics
    // FIXME(generic_arg_infer): this error message should say in the return type or sth like that.
    [0; 3]
}

fn main() {
    foo();
}

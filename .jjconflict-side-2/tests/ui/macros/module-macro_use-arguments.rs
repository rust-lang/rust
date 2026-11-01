//@ reference: macro.decl.scope.macro_use.syntax
#[macro_use(foo, bar)] //~ ERROR arguments to `macro_use` are not allowed here
mod foo {
}

fn main() {
}

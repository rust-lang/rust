#![feature(register_attr)]
#![feature(register_tool)]

#![register_attr(attr)]
#![register_tool(tool)]

#[no_implicit_prelude]
mod m {
    #[attr] //~ ERROR cannot find attribute `attr` in this scope
    #[tool::attr] //~ ERROR failed to resolve: use of undeclared type or module `tool`
    fn check() {}
}

fn main() {}

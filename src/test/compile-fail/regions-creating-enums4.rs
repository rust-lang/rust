enum ast {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_bad2(x: &a/ast, y: &a/ast, z: &ast) -> ast {
    add(x, y) //~ ERROR mismatched types: expected `ast/&` but found `ast/&a`
}

fn main() {
}
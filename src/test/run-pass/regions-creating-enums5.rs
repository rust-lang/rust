enum ast/& {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_ok(x: &a.ast, y: &a.ast, z: &ast) -> ast/&a {
    add(x, y)
}

fn main() {
}
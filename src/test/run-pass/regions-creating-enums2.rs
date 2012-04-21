enum ast/& {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_ok(x: &ast, y: &ast) -> ast {
    add(x, y)
}

fn main() {
}
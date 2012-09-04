enum ast {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_ok(x: &r/ast, y: &r/ast) -> ast/&r {
    add(x, y)
}

fn main() {
}
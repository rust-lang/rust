enum ast/& {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_bad1(x: &a.ast, y: &b.ast) -> ast/&a {
    add(x, y) //! ERROR mismatched types: expected `&a.ast/&a` but found `&b.ast/&b`
}

fn main() {
}
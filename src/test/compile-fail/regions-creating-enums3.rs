enum ast {
    num(uint),
    add(&ast, &ast)
}

fn mk_add_bad1(x: &a/ast, y: &b/ast) -> ast/&a {
    add(x, y) //~ ERROR cannot infer an appropriate lifetime
        //~^ ERROR cannot infer an appropriate lifetime
}

fn main() {
}

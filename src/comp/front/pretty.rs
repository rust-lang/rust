use std;

fn unknown() -> str {
    ret "<unknown ast node>";
}

fn print_expr(@ast.expr expr) -> str {
    alt (expr.node) {
        case (ast.expr_lit(?lit, _)) {
            ret print_expr_lit(lit);
        }
        case (ast.expr_binary(?op, ?lhs, ?rhs, _)) {
            ret print_expr_binary(op, lhs, rhs);
        }
        case (_) {
            ret unknown();
        }
    }
}

fn print_expr_lit(@ast.lit lit) -> str {
    alt (lit.node) {
        case (ast.lit_str(?s)) {
            ret "\"" + s + "\"";
        }
        case (_) {
            ret unknown();
        }
    }
}

fn print_expr_binary(ast.binop op, @ast.expr lhs, @ast.expr rhs) -> str {
    alt (op) {
        case (ast.add) {
            auto l = print_expr(lhs);
            auto r = print_expr(rhs);
            ret l + " + " + r;
        }
    }
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C ../.. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

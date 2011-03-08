import std._int;
import std._str;
import std._uint;
import std._vec;

export print_expr;

// FIXME this is superseded by ../pretty/pprust.rs. can it be dropped?

fn unknown() -> str {
    ret "<unknown ast node>";
}

fn print_expr(@ast.expr expr) -> str {
    alt (expr.node) {
        case (ast.expr_lit(?lit, _)) {
            ret print_lit(lit);
        }
        case (ast.expr_binary(?op, ?lhs, ?rhs, _)) {
            ret print_expr_binary(op, lhs, rhs);
        }
        case (ast.expr_call(?path, ?args, _)) {
            ret print_expr_call(path, args);
        }
        case (ast.expr_path(?path, _, _)) {
            ret print_path(path);
        }
        case (_) {
            ret unknown();
        }
    }
}

fn print_lit(@ast.lit lit) -> str {
    alt (lit.node) {
        case (ast.lit_str(?s)) {
            ret "\"" + s + "\"";
        }
        case (ast.lit_int(?i)) {
            ret _int.to_str(i, 10u);
        }
        case (ast.lit_uint(?u)) {
            ret _uint.to_str(u, 10u);
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

fn print_expr_call(@ast.expr path_expr, vec[@ast.expr] args) -> str {
    auto s = print_expr(path_expr);

    s += "(";
    fn print_expr_ref(&@ast.expr e) -> str { ret print_expr(e); }
    auto mapfn = print_expr_ref;
    auto argstrs = _vec.map[@ast.expr, str](mapfn, args);
    s += _str.connect(argstrs, ", ");
    s += ")";

    ret s;
}

fn print_path(ast.path path) -> str {
    ret _str.connect(path.node.idents, ".");
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

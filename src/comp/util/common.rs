
import std::map;
import std::map::hashmap;
import std::uint;
import std::int;
import std::vec;
import std::option;
import std::option::none;
import std::option::some;
import front::ast;
import front::ast::ty;
import front::ast::pat;
import front::codemap::codemap;
import front::ast::lit;
import front::ast::path;
import middle::walk;
import std::io::stdout;
import std::io::str_writer;
import std::io::string_writer;
import pretty::pprust::print_block;
import pretty::pprust::print_item;
import pretty::pprust::print_expr;
import pretty::pprust::print_path;
import pretty::pprust::print_decl;
import pretty::pprust::print_fn;
import pretty::pprust::print_type;
import pretty::ppaux::print_literal;
import pretty::ppaux::mo_untyped;
import pretty::pp::mk_printer;

type filename = str;

type span = rec(uint lo, uint hi);

type spanned[T] = rec(T node, span span);

type flag = hashmap[str, ()];

tag ty_mach {
    ty_i8;
    ty_i16;
    ty_i32;
    ty_i64;
    ty_u8;
    ty_u16;
    ty_u32;
    ty_u64;
    ty_f32;
    ty_f64;
}

tag ty_or_bang[T] { a_ty(T); a_bang; }

fn ty_mach_to_str(ty_mach tm) -> str {
    alt (tm) {
        case (ty_u8) { ret "u8"; }
        case (ty_u16) { ret "u16"; }
        case (ty_u32) { ret "u32"; }
        case (ty_u64) { ret "u64"; }
        case (ty_i8) { ret "i8"; }
        case (ty_i16) { ret "i16"; }
        case (ty_i32) { ret "i32"; }
        case (ty_i64) { ret "i64"; }
        case (ty_f32) { ret "f32"; }
        case (ty_f64) { ret "f64"; }
    }
}

fn new_str_hash[V]() -> std::map::hashmap[str, V] {
    let std::map::hashfn[str] hasher = std::str::hash;
    let std::map::eqfn[str] eqer = std::str::eq;
    ret std::map::mk_hashmap[str, V](hasher, eqer);
}

fn def_eq(&ast::def_id a, &ast::def_id b) -> bool {
    ret a._0 == b._0 && a._1 == b._1;
}

fn hash_def(&ast::def_id d) -> uint {
    auto h = 5381u;
    h = (h << 5u) + h ^ (d._0 as uint);
    h = (h << 5u) + h ^ (d._1 as uint);
    ret h;
}

fn new_def_hash[V]() -> std::map::hashmap[ast::def_id, V] {
    let std::map::hashfn[ast::def_id] hasher = hash_def;
    let std::map::eqfn[ast::def_id] eqer = def_eq;
    ret std::map::mk_hashmap[ast::def_id, V](hasher, eqer);
}

fn new_int_hash[V]() -> std::map::hashmap[int, V] {
    fn hash_int(&int x) -> uint { ret x as uint; }
    fn eq_int(&int a, &int b) -> bool { ret a == b; }
    auto hasher = hash_int;
    auto eqer = eq_int;
    ret std::map::mk_hashmap[int, V](hasher, eqer);
}

fn new_uint_hash[V]() -> std::map::hashmap[uint, V] {
    fn hash_uint(&uint x) -> uint { ret x; }
    fn eq_uint(&uint a, &uint b) -> bool { ret a == b; }
    auto hasher = hash_uint;
    auto eqer = eq_uint;
    ret std::map::mk_hashmap[uint, V](hasher, eqer);
}

fn new_seq_hash[K, V](fn(&K) -> uint key_idx,
                      fn(&uint) -> K idx_key) -> std::map::hashmap[K, V] {
    fn ensure_size[V](&mutable vec[mutable option::t[V]] bkts, uint idx) {
        auto bkt_len = vec::len(bkts);
        if (idx >= bkt_len) {
            auto needed = idx - bkt_len + 1u;
            auto new = vec::init_elt_mut(option::none[V], needed);
            bkts += new;
        }
    }

    obj seq_hash[K, V](mutable uint nelts,
                       mutable vec[mutable option::t[V]] bkts,
                       fn(&K) -> uint key_idx,
                       fn(&uint) -> K idx_key) {

        fn size() -> uint { nelts }

        fn insert(&K key, &V value) -> bool {
            auto idx = key_idx(key);
            ensure_size(bkts, idx);
            if (option::is_some(bkts.(idx))) {
                bkts.(idx) = option::some(value);
                ret false;
            } else {
                bkts.(idx) = option::some(value);
                nelts += 1u;
                ret true;
            }
        }

        fn contains_key(&K key) -> bool {
            auto idx = key_idx(key);
            ensure_size(bkts, idx);
            if (option::is_some(bkts.(idx))) {
                ret true;
            } else {
                ret false;
            }
        }

        fn get(&K key) -> V {
            ret alt (self.find(key)) {
                case (option::some(?v)) { v }
                case (_) { fail }
            };
        }

        fn find(&K key) -> option::t[V] {
            auto idx = key_idx(key);
            ensure_size(bkts, idx);
            ret bkts.(idx);
        }

        fn remove(&K key) -> option::t[V] {
            auto idx = key_idx(key);
            ensure_size(bkts, idx);
            auto val = bkts.(idx);
            if (option::is_some(val)) {
                bkts.(idx) = option::none;
                nelts -= 1u;
            }
            ret val;
        }

        fn rehash() { }

        iter items() -> @tup(K, V) {
            auto idx = 0u;
            auto bktsize = vec::len(bkts);
            while (idx < bktsize) {
                alt (bkts.(idx)) {
                    case (option::some(?v)) {
                        // FIXME: Appease alias analysis
                        auto value = v;
                        put @tup(idx_key(idx), value);
                    }
                    case (_) { }
                }
                idx += 1u;
            }
        }
    }

    let vec[mutable option::t[V]] bkts = [mutable];
    ret seq_hash[K, V](0u, bkts, key_idx, idx_key);
}

// A specialized map for keys that are sequential ints
fn new_seq_int_hash[V]() -> std::map::hashmap[int, V] {
    auto key_idx = fn(&int key) -> uint { key as uint };
    auto idx_key = fn(&uint idx) -> int { idx as int };
    ret new_seq_hash(key_idx, idx_key);
}

// A specialized map for keys that are sequential uints
fn new_seq_uint_hash[V]() -> std::map::hashmap[uint, V] {
    auto key_idx = fn(&uint key) -> uint { key };
    auto idx_key = fn(&uint idx) -> uint { idx };
    ret new_seq_hash(key_idx, idx_key);
}

fn istr(int i) -> str { ret int::to_str(i, 10u); }

fn uistr(uint i) -> str { ret uint::to_str(i, 10u); }

fn elt_expr(&ast::elt e) -> @ast::expr { ret e.expr; }

fn elt_exprs(&vec[ast::elt] elts) -> vec[@ast::expr] {
    auto f = elt_expr;
    ret vec::map[ast::elt, @ast::expr](f, elts);
}

fn field_expr(&ast::field f) -> @ast::expr { ret f.node.expr; }

fn field_exprs(vec[ast::field] fields) -> vec[@ast::expr] {
    auto f = field_expr;
    ret vec::map[ast::field, @ast::expr](f, fields);
}

fn log_expr(&ast::expr e) { log pretty::pprust::expr_to_str(@e); }

fn log_expr_err(&ast::expr e) { log_err pretty::pprust::expr_to_str(@e); }

fn log_ty_err(&ty t) { log_err pretty::pprust::ty_to_str(t); }

fn log_pat_err(&@pat p) { log_err pretty::pprust::pat_to_str(p); }

fn log_block(&ast::block b) { log pretty::pprust::block_to_str(b); }

fn log_block_err(&ast::block b) { log_err pretty::pprust::block_to_str(b); }

fn log_item_err(&@ast::item i) { log_err pretty::pprust::item_to_str(i); }

fn log_fn(&ast::_fn f, str name, vec[ast::ty_param] params) {
    log pretty::pprust::fun_to_str(f, name, params);
}

fn log_fn_err(&ast::_fn f, str name, vec[ast::ty_param] params) {
    log_err pretty::pprust::fun_to_str(f, name, params);
}

fn log_stmt(&ast::stmt st) { log pretty::pprust::stmt_to_str(st); }

fn log_stmt_err(&ast::stmt st) { log_err pretty::pprust::stmt_to_str(st); }

fn has_nonlocal_exits(&ast::block b) -> bool {
    auto has_exits = @mutable false;
    fn visit_expr(@mutable bool flag, &@ast::expr e) {
        alt (e.node) {
            case (ast::expr_break) { *flag = true; }
            case (ast::expr_cont) { *flag = true; }
            case (_) { }
        }
    }
    auto v =
        rec(visit_expr_pre=bind visit_expr(has_exits, _)
            with walk::default_visitor());
    walk::walk_block(v, b);
    ret *has_exits;
}

fn local_rhs_span(&@ast::local l, &span def) -> span {
    alt (l.node.init) {
        case (some(?i)) { ret i.expr.span; }
        case (_) { ret def; }
    }
}

fn lit_eq(&@ast::lit l, &@ast::lit m) -> bool {
    alt (l.node) {
        case (ast::lit_str(?s, ?kind_s)) {
            alt (m.node) {
                case (ast::lit_str(?t, ?kind_t)) {
                    ret s == t && kind_s == kind_t;
                }
                case (_) { ret false; }
            }
        }
        case (ast::lit_char(?c)) {
            alt (m.node) {
                case (ast::lit_char(?d)) { ret c == d; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_int(?i)) {
            alt (m.node) {
                case (ast::lit_int(?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_uint(?i)) {
            alt (m.node) {
                case (ast::lit_uint(?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_mach_int(_, ?i)) {
            alt (m.node) {
                case (ast::lit_mach_int(_, ?j)) { ret i == j; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_float(?s)) {
            alt (m.node) {
                case (ast::lit_float(?t)) { ret s == t; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_mach_float(_, ?s)) {
            alt (m.node) {
                case (ast::lit_mach_float(_, ?t)) { ret s == t; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_nil) {
            alt (m.node) {
                case (ast::lit_nil) { ret true; }
                case (_) { ret false; }
            }
        }
        case (ast::lit_bool(?b)) {
            alt (m.node) {
                case (ast::lit_bool(?c)) { ret b == c; }
                case (_) { ret false; }
            }
        }
    }
}

fn respan[T](&span sp, &T t) -> spanned[T] { ret rec(node=t, span=sp); }

fn may_begin_ident(char c) -> bool { ret is_alpha(c) || c == '_'; }

fn in_range(char c, char lo, char hi) -> bool { ret lo <= c && c <= hi; }

fn is_alpha(char c) -> bool {
    ret in_range(c, 'a', 'z') || in_range(c, 'A', 'Z');
}

fn is_dec_digit(char c) -> bool { ret in_range(c, '0', '9'); }

fn is_alnum(char c) -> bool { ret is_alpha(c) || is_dec_digit(c); }

fn is_hex_digit(char c) -> bool {
    ret in_range(c, '0', '9') || in_range(c, 'a', 'f') ||
            in_range(c, 'A', 'F');
}

fn is_bin_digit(char c) -> bool { ret c == '0' || c == '1'; }

// FIXME move to vec
fn any[T](&fn(&T) -> bool f, &vec[T] v) -> bool {
    for (T t in v) {
        if (f(t)) { ret true; } 
    }
    ret false;
}

//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:
//

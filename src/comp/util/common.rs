import std._uint;
import std._int;

type pos = rec(uint line, uint col);
type span = rec(str filename, pos lo, pos hi);
type spanned[T] = rec(T node, span span);

// FIXME: import std.util.option and use it here.
// import std.util.option;

tag option[T] {
  none;
  some(T);
}

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

fn new_str_hash[V]() -> std.map.hashmap[str,V] {
    let std.map.hashfn[str] hasher = std._str.hash;
    let std.map.eqfn[str] eqer = std._str.eq;
    ret std.map.mk_hashmap[str,V](hasher, eqer);
}

fn istr(int i) -> str {
    ret _int.to_str(i, 10u);
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

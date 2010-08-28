
type ty_mach = tag( ty_i8(), ty_i16(), ty_i32(), ty_i64(),
                    ty_u8(), ty_u16(), ty_u32(), ty_u64(),
                    ty_f32(), ty_f16() );

fn ty_mach_to_str(ty_mach tm) -> str {
    alt (tm) {
        case (ty_u8()) { ret "u8"; }
        case (ty_i8()) { ret "i8"; }
        case (ty_u16()) { ret "u16"; }
        case (ty_i16()) { ret "i16"; }
        case (ty_u32()) { ret "u32"; }
        case (ty_i32()) { ret "i32"; }
        case (ty_u64()) { ret "u64"; }
        case (ty_i64()) { ret "i64"; }
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

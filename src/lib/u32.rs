/*
Module: u32
*/

/*
Function: min_value

Return the minimal value for a u32
*/
pure fn min_value() -> u32 { ret 0u32; }

/*
Function: max_value

Return the maximal value for a u32
*/
pure fn max_value() -> u32 { ret 4294967296u32; }

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

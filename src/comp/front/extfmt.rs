/* The 'fmt' extension is modeled on the posix printf system.
 * 
 * A posix conversion ostensibly looks like this:
 * 
 * %[parameter][flags][width][.precision][length]type
 * 
 * Given the different numeric type bestiary we have, we omit the 'length'
 * parameter and support slightly different conversions for 'type':
 * 
 * %[parameter][flags][width][.precision]type
 * 
 * we also only support translating-to-rust a tiny subset of the possible
 * combinations at the moment.
 */

use std;

import std.option;

tag signedness {
    signed;
    unsigned;
}

tag caseness {
    case_upper;
    case_lower;
}

tag ty {
    ty_bool;
    ty_str;
    ty_char;
    ty_int(signedness);
    ty_bits;
    ty_hex(caseness);
    // FIXME: More types
}

tag flag {
    flag_left_justify;
    flag_left_zero_pad;
    flag_left_space_pad;
    flag_plus_if_positive;
    flag_alternate;
}

tag count {
    count_is(int);
    count_is_param(int);
    count_is_next_param;
    count_implied;
}

// A formatted conversion from an expression to a string
tag conv {
    conv_param(option.t[int]);
    conv_flags(vec[flag]);
    conv_width(count);
    conv_precision(count);
    conv_ty(ty);
}

// A fragment of the output sequence
tag piece {
    piece_string(str);
    piece_conv(str);
}

fn expand_syntax_ext(vec[@ast.expr] args,
                     option.t[@ast.expr] body) -> @ast.expr {
    fail;
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

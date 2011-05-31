import std::io;
import std::str;
import std::vec;
import std::int;
import std::map;
import std::map::hashmap;
import std::option;
import std::option::some;
import std::option::none;
import driver::session::session;
import util::common;
import util::common::new_str_hash;
import util::data::interner;

state type reader = state obj {
    fn is_eof() -> bool;
    fn curr() -> char;
    fn next() -> char;
    fn init();
    fn bump();
    fn mark();
    fn get_mark_chpos() -> uint;
    fn get_interner() -> @interner::interner[str];
    fn get_chpos() -> uint;
    fn get_filemap() -> codemap::filemap;
    fn err(str m);
};

fn new_reader(session sess, io::reader rdr,
              codemap::filemap filemap,
              @interner::interner[str] itr) -> reader {

    state obj reader(session sess,
                     str file,
                     uint len,
                     mutable uint pos,
                     mutable char ch,
                     mutable uint mark_chpos,
                     mutable uint chpos,
                     mutable vec[str] strs,
                     codemap::filemap fm,
                     @interner::interner[str] itr) {

        fn is_eof() -> bool {
            ret ch == -1 as char;
        }

        fn mark() { mark_chpos = chpos; }
        fn get_mark_chpos() -> uint { ret mark_chpos; }
        fn get_chpos() -> uint { ret chpos; }

        fn curr() -> char {
            ret ch;
        }

        fn next() -> char {
            if (pos < len) {ret str::char_at(file, pos);}
            else {ret -1 as char;}
        }

        fn init() {
            if (pos < len) {
                auto next = str::char_range_at(file, pos);
                pos = next._1;
                ch = next._0;
            }
        }

        fn bump() {
            if (pos < len) {
                chpos += 1u;
                if (ch == '\n') {
                    codemap::next_line(fm, chpos);
                }
                auto next = str::char_range_at(file, pos);
                pos = next._1;
                ch = next._0;
            } else {
                ch = -1 as char;
            }
        }

        fn get_interner() -> @interner::interner[str] { ret itr; }

        fn get_filemap() -> codemap::filemap {
            ret fm;
        }

        fn err(str m) {
            sess.span_err(rec(lo=chpos, hi=chpos), m);
        }
    }
    auto file = str::unsafe_from_bytes(rdr.read_whole_stream());
    let vec[str] strs = [];
    auto rd = reader(sess, file, str::byte_len(file), 0u, -1 as char,
                     filemap.start_pos, filemap.start_pos,
                     strs, filemap, itr);
    rd.init();
    ret rd;
}

fn in_range(char c, char lo, char hi) -> bool {
    ret lo <= c && c <= hi;
}

fn is_alpha(char c) -> bool {
    ret in_range(c, 'a', 'z') ||
        in_range(c, 'A', 'Z');
}

fn is_dec_digit(char c) -> bool {
    ret in_range(c, '0', '9');
}

fn is_alnum(char c) -> bool {
    ret is_alpha(c) || is_dec_digit(c);
}

fn is_hex_digit(char c) -> bool {
    ret in_range(c, '0', '9') ||
        in_range(c, 'a', 'f') ||
        in_range(c, 'A', 'F');
}

fn is_bin_digit(char c) -> bool {
    ret c == '0' || c == '1';
}

fn dec_digit_val(char c) -> int {
    ret (c as int) - ('0' as int);
}

fn hex_digit_val(char c) -> int {
    if (in_range(c, '0', '9')) {
        ret (c as int) - ('0' as int);
    }

    if (in_range(c, 'a', 'f')) {
        ret ((c as int) - ('a' as int)) + 10;
    }

    if (in_range(c, 'A', 'F')) {
        ret ((c as int) - ('A' as int)) + 10;
    }

    fail;
}

fn bin_digit_value(char c) -> int {
    if (c == '0') { ret 0; }
    ret 1;
}

fn is_whitespace(char c) -> bool {
    ret c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

fn consume_any_whitespace(&reader rdr) {
    while (is_whitespace(rdr.curr())) {
        rdr.bump();
    }
    be consume_any_line_comment(rdr);
}

fn consume_any_line_comment(&reader rdr) {
    if (rdr.curr() == '/') {
        alt (rdr.next()) {
            case ('/') {
                while (rdr.curr() != '\n' && !rdr.is_eof()) {
                    rdr.bump();
                }
                // Restart whitespace munch.
                be consume_any_whitespace(rdr);
            }
            case ('*') {
                rdr.bump();
                rdr.bump();
                be consume_block_comment(rdr);
            }
            case (_) {
                ret;
            }
        }
    }
}


fn consume_block_comment(&reader rdr) {
    let int level = 1;
    while (level > 0) {
        if (rdr.is_eof()) {
            rdr.err("unterminated block comment");
            fail;
        }
        if (rdr.curr() == '/' && rdr.next() == '*') {
            rdr.bump();
            rdr.bump();
            level += 1;
        } else {
            if (rdr.curr() == '*' && rdr.next() == '/') {
                rdr.bump();
                rdr.bump();
                level -= 1;
            } else {
                rdr.bump();
            }
        }
    }
    // restart whitespace munch.
    be consume_any_whitespace(rdr);
}

fn digits_to_string(str s) -> int {

    let int accum_int = 0;
    let int i = 0;

    for (u8 c in s) {
        accum_int *= 10;
        accum_int += dec_digit_val(c as char);
    }

    ret accum_int;
}

fn scan_exponent(&reader rdr) -> option::t[str] {
    auto c = rdr.curr();
    auto res = "";

    if (c == 'e' || c == 'E') {
        res += str::from_bytes([c as u8]);
        rdr.bump();
        c = rdr.curr();
        if (c == '-' || c == '+') {
            res += str::from_bytes([c as u8]);
            rdr.bump();
        }
        auto exponent = scan_dec_digits(rdr);
        if (str::byte_len(exponent) > 0u) {
            ret(some(res + exponent));
        }
        else {
            rdr.err("scan_exponent: bad fp literal");
            fail;
        }
    }
    else {
        ret none[str];
    }
}

fn scan_dec_digits(&reader rdr) -> str {

    auto c = rdr.curr();
    let str res = "";

    while (is_dec_digit (c) || c == '_') {
        if (c != '_') {
            res += str::from_bytes([c as u8]);
        }
        rdr.bump();
        c = rdr.curr();
    }

    ret res;
}

fn scan_number(char c, &reader rdr) -> token::token {
    auto accum_int = 0;
    let str dec_str = "";
    let bool is_dec_integer = false;
    auto n = rdr.next();

    if (c == '0' && n == 'x') {
        rdr.bump();
        rdr.bump();
        c = rdr.curr();
        while (is_hex_digit(c) || c == '_') {
            if (c != '_') {
                accum_int *= 16;
                accum_int += hex_digit_val(c);
            }
            rdr.bump();
            c = rdr.curr();
        }
    } else if (c == '0' && n == 'b') {
        rdr.bump();
        rdr.bump();
        c = rdr.curr();
        while (is_bin_digit(c) || c == '_') {
            if (c != '_') {
                accum_int *= 2;
                accum_int += bin_digit_value(c);
            }
            rdr.bump();
            c = rdr.curr();
        }
    } else {
        dec_str = scan_dec_digits(rdr);
        is_dec_integer = true;
    }

    if (is_dec_integer) {
        accum_int = digits_to_string(dec_str);
    }

    c = rdr.curr();
    n = rdr.next();

    if (c == 'u' || c == 'i') {
        let bool signed = (c == 'i');
        rdr.bump();
        c = rdr.curr();
        if (c == '8') {
            rdr.bump();
            if (signed) {
                ret token::LIT_MACH_INT(common::ty_i8, accum_int);
            } else {
                ret token::LIT_MACH_INT(common::ty_u8, accum_int);
            }
        }

        n = rdr.next();
        if (c == '1' && n == '6') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token::LIT_MACH_INT(common::ty_i16, accum_int);
            } else {
                ret token::LIT_MACH_INT(common::ty_u16, accum_int);
            }
        }
        if (c == '3' && n == '2') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token::LIT_MACH_INT(common::ty_i32, accum_int);
            } else {
                ret token::LIT_MACH_INT(common::ty_u32, accum_int);
            }
        }

        if (c == '6' && n == '4') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token::LIT_MACH_INT(common::ty_i64, accum_int);
            } else {
                ret token::LIT_MACH_INT(common::ty_u64, accum_int);
            }
        }

        if (signed) {
            ret token::LIT_INT(accum_int);
        } else {
            // FIXME: should cast in the target bit-width.
            ret token::LIT_UINT(accum_int as uint);
        }
    }
    c = rdr.curr();

    if (c == '.') {
        // Parse a floating-point number.
        rdr.bump();
        auto dec_part = scan_dec_digits(rdr);
        auto float_str = dec_str + "." + dec_part;
        c = rdr.curr();
        auto exponent_str = scan_exponent(rdr);
        alt (exponent_str) {
            case (some(?s)) {
                float_str += s;
            }
            case (none) {
            }
        }

        c = rdr.curr();
        if (c == 'f') {
            rdr.bump();
            c = rdr.curr();
            n = rdr.next();
            if (c == '3' && n == '2') {
                rdr.bump(); rdr.bump();
                ret token::LIT_MACH_FLOAT(util::common::ty_f32,
                    interner::intern[str](*rdr.get_interner(), float_str));
            }
            else if (c == '6' && n == '4') {
                rdr.bump(); rdr.bump();
                ret token::LIT_MACH_FLOAT(util::common::ty_f64,
                    interner::intern[str](*rdr.get_interner(), float_str));
                /* FIXME: if this is out of range for either a 32-bit or
                   64-bit float, it won't be noticed till the back-end */
            }
        }
        else {
            ret token::LIT_FLOAT(interner::intern[str](*rdr.get_interner(),
                                                       float_str));
        }
    }

    auto maybe_exponent = scan_exponent(rdr);
    alt(maybe_exponent) {
        case(some(?s)) {
            ret token::LIT_FLOAT(interner::intern[str](*rdr.get_interner(),
                                                       dec_str + s));
        }
        case(none) {
            ret token::LIT_INT(accum_int);
        }
    }
}

fn scan_numeric_escape(&reader rdr, uint n_hex_digits) -> char {
    auto accum_int = 0;
    while (n_hex_digits != 0u) {
        auto n = rdr.curr();
        rdr.bump();
        if (!is_hex_digit(n)) {
            rdr.err(#fmt("illegal numeric character escape: %d", n as int));
            fail;
        }
        accum_int *= 16;
        accum_int += hex_digit_val(n);
        n_hex_digits -= 1u;
    }
    ret accum_int as char;
}


fn next_token(&reader rdr) -> token::token {
    auto accum_str = "";

    consume_any_whitespace(rdr);

    if (rdr.is_eof()) { ret token::EOF; }

    rdr.mark();
    auto c = rdr.curr();

    if (is_alpha(c) || c == '_') {
        while (is_alnum(c) || c == '_') {
            str::push_char(accum_str, c);
            rdr.bump();
            c = rdr.curr();
        }

        if (str::eq(accum_str, "_")) {
            ret token::UNDERSCORE;
        }

        auto is_mod_name = c == ':' && rdr.next() == ':';
        ret token::IDENT(interner::intern[str](*rdr.get_interner(),
                                               accum_str), is_mod_name);
    }

    if (is_dec_digit(c)) {
        ret scan_number(c, rdr);
    }

    fn binop(&reader rdr, token::binop op) -> token::token {
        rdr.bump();
        if (rdr.curr() == '=') {
            rdr.bump();
            ret token::BINOPEQ(op);
        } else {
            ret token::BINOP(op);
        }
    }

    alt (c) {
        // One-byte tokens.
        case ('?') { rdr.bump(); ret token::QUES; }
        case (';') { rdr.bump(); ret token::SEMI; }
        case (',') { rdr.bump(); ret token::COMMA; }
        case ('.') { rdr.bump(); ret token::DOT; }
        case ('(') { rdr.bump(); ret token::LPAREN; }
        case (')') { rdr.bump(); ret token::RPAREN; }
        case ('{') { rdr.bump(); ret token::LBRACE; }
        case ('}') { rdr.bump(); ret token::RBRACE; }
        case ('[') { rdr.bump(); ret token::LBRACKET; }
        case (']') { rdr.bump(); ret token::RBRACKET; }
        case ('@') { rdr.bump(); ret token::AT; }
        case ('#') { rdr.bump(); ret token::POUND; }
        case ('~') { rdr.bump(); ret token::TILDE; }
        case (':') {
            rdr.bump();
            if (rdr.curr() == ':') {
                rdr.bump();
                ret token::MOD_SEP;
            }
            else {
                ret token::COLON;
            };
        }

        // Multi-byte tokens.
        case ('=') {
            rdr.bump();
            if (rdr.curr() == '=') {
                rdr.bump();
                ret token::EQEQ;
            } else {
                ret token::EQ;
            }
        }

        case ('!') {
            rdr.bump();
            if (rdr.curr() == '=') {
                rdr.bump();
                ret token::NE;
            } else {
                ret token::NOT;
            }
        }

        case ('<') {
            rdr.bump();
            alt (rdr.curr()) {
                case ('=') {
                    rdr.bump();
                    ret token::LE;
                }
                case ('<') {
                    ret binop(rdr, token::LSL);
                }
                case ('-') {
                    rdr.bump();
                    ret token::LARROW;
                }
                case ('|') {
                    rdr.bump();
                    ret token::SEND;
                }
                case (_) {
                    ret token::LT;
                }
            }
        }

        case ('>') {
            rdr.bump();
            alt (rdr.curr()) {
                case ('=') {
                    rdr.bump();
                    ret token::GE;
                }

                case ('>') {
                    if (rdr.next() == '>') {
                        rdr.bump();
                        ret binop(rdr, token::ASR);
                    } else {
                        ret binop(rdr, token::LSR);
                    }
                }

                case (_) {
                    ret token::GT;
                }
            }
        }

        case ('\'') {
            rdr.bump();
            auto c2 = rdr.curr();
            rdr.bump();
            if (c2 == '\\') {
                auto escaped = rdr.curr();
                rdr.bump();
                alt (escaped) {
                    case ('n') { c2 = '\n'; }
                    case ('r') { c2 = '\r'; }
                    case ('t') { c2 = '\t'; }
                    case ('\\') { c2 = '\\'; }
                    case ('\'') { c2 = '\''; }

                    case ('x') { c2 = scan_numeric_escape(rdr, 2u); }
                    case ('u') { c2 = scan_numeric_escape(rdr, 4u); }
                    case ('U') { c2 = scan_numeric_escape(rdr, 8u); }

                    case (?c2) {
                        rdr.err(#fmt("unknown character escape: %d",
                                     c2 as int));
                        fail;
                    }
                }
            }

            if (rdr.curr() != '\'') {
                rdr.err("unterminated character constant");
                fail;
            }
            rdr.bump(); // advance curr past token
            ret token::LIT_CHAR(c2);
        }

        case ('"') {
            rdr.bump();
            while (rdr.curr() != '"') {
                auto ch = rdr.curr();
                rdr.bump();
                alt (ch) {
                    case ('\\') {
                        auto escaped = rdr.curr();
                        rdr.bump();
                        alt (escaped) {
                            case ('n') {
                                str::push_byte(accum_str, '\n' as u8);
                            }
                            case ('r') {
                                str::push_byte(accum_str, '\r' as u8);
                            }
                            case ('t') {
                                str::push_byte(accum_str, '\t' as u8);
                            }
                            case ('\\') {
                                str::push_byte(accum_str, '\\' as u8);
                            }
                            case ('"') {
                                str::push_byte(accum_str, '"' as u8);
                            }

                            case ('x') {
                                str::push_char(accum_str,
                                               scan_numeric_escape(rdr, 2u));
                            }

                            case ('u') {
                                str::push_char(accum_str,
                                               scan_numeric_escape(rdr, 4u));
                            }

                            case ('U') {
                                str::push_char(accum_str,
                                               scan_numeric_escape(rdr, 8u));
                            }

                            case (?c2) {
                                rdr.err(#fmt("unknown string escape: %d",
                                             c2 as int));
                                fail;
                            }
                        }
                    }
                    case (_) {
                        str::push_char(accum_str, ch);
                    }
                }
            }
            rdr.bump();
            ret token::LIT_STR(interner::intern[str](*rdr.get_interner(),
                                                     accum_str));
        }

        case ('-') {
            if (rdr.next() == '>') {
                rdr.bump();
                rdr.bump();
                ret token::RARROW;
            } else {
                ret binop(rdr, token::MINUS);
            }
        }

        case ('&') {
            if (rdr.next() == '&') {
                rdr.bump();
                rdr.bump();
                ret token::ANDAND;
            } else {
                ret binop(rdr, token::AND);
            }
        }

        case ('|') {
            alt (rdr.next()) {
                case ('|') {
                    rdr.bump();
                    rdr.bump();
                    ret token::OROR;
                }
                case ('>') {
                    rdr.bump();
                    rdr.bump();
                    ret token::RECV;
                }
                case (_) {
                    ret binop(rdr, token::OR);
                }
            }
        }

        case ('+') {
            ret binop(rdr, token::PLUS);
        }

        case ('*') {
            ret binop(rdr, token::STAR);
        }

        case ('/') {
            ret binop(rdr, token::SLASH);
        }

        case ('^') {
            ret binop(rdr, token::CARET);
        }

        case ('%') {
            ret binop(rdr, token::PERCENT);
        }

        case (?c) {
            rdr.err(#fmt("unkown start of token: %d", c as int));
            fail;
        }
    }

    fail;
}

tag cmnt_ {
    cmnt_line(str);
    cmnt_block(vec[str]);
}

type cmnt = rec(cmnt_ val, uint pos, bool space_after);

fn consume_whitespace(&reader rdr) -> uint {
    auto lines = 0u;
    while (is_whitespace(rdr.curr())) {
        if (rdr.curr() == '\n') {lines += 1u;}
        rdr.bump();
    }
    ret lines;
}

fn read_line_comment(&reader rdr) -> cmnt {
    auto p = rdr.get_chpos();
    rdr.bump(); rdr.bump();
    while (rdr.curr() == ' ') {rdr.bump();}
    auto val = "";
    while (rdr.curr() != '\n' && !rdr.is_eof()) {
        str::push_char(val, rdr.curr());
        rdr.bump();
    }
    ret rec(val=cmnt_line(val),
            pos=p,
            space_after=consume_whitespace(rdr) > 1u);
}

fn read_block_comment(&reader rdr) -> cmnt {
    auto p = rdr.get_chpos();
    rdr.bump(); rdr.bump();
    while (rdr.curr() == ' ') {rdr.bump();}
    let vec[str] lines = [];
    auto val = "";
    auto level = 1;
    while (true) {
        if (rdr.curr() == '\n') {
            vec::push[str](lines, val);
            val = "";
            consume_whitespace(rdr);
        } else {
            if (rdr.curr() == '*' && rdr.next() == '/') {
                level -= 1;
                if (level == 0) {
                    rdr.bump(); rdr.bump();
                    vec::push[str](lines, val);
                    break;
                }
            } else if (rdr.curr() == '/' && rdr.next() == '*') {
                level += 1;
            }
            str::push_char(val, rdr.curr());
            rdr.bump();
        }
        if (rdr.is_eof()) {
            rdr.err("Unexpected end of file in block comment");
            fail;
        }
    }
    ret rec(val=cmnt_block(lines),
            pos=p,
            space_after=consume_whitespace(rdr) > 1u);
}

fn gather_comments(session sess, str path) -> vec[cmnt] {
    auto srdr = io::file_reader(path);
    auto itr = @interner::mk[str](str::hash, str::eq);
    auto rdr = new_reader(sess, srdr, codemap::new_filemap(path, 0u), itr);
    let vec[cmnt] comments = [];
    while (!rdr.is_eof()) {
        while (true) {
            consume_whitespace(rdr);
            if (rdr.curr() == '/' && rdr.next() == '/') {
                vec::push[cmnt](comments, read_line_comment(rdr));
            } else if (rdr.curr() == '/' && rdr.next() == '*') {
                vec::push[cmnt](comments, read_block_comment(rdr));
            } else { break; }
        }
        next_token(rdr);
    }
    ret comments;
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

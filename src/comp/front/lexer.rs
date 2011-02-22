import std.io.stdio_reader;
import std._str;
import std.map;
import std.map.hashmap;
import util.common;
import util.common.new_str_hash;

state type reader = state obj {
                          fn is_eof() -> bool;
                          fn curr() -> char;
                          fn next() -> char;
                          impure fn bump();
                          fn mark();
                          fn get_filename() -> str;
                          fn get_mark_pos() -> common.pos;
                          fn get_curr_pos() -> common.pos;
                          fn get_keywords() -> hashmap[str,token.token];
                          fn get_reserved() -> hashmap[str,()];
};

fn new_reader(stdio_reader rdr, str filename) -> reader
{
    state obj reader(stdio_reader rdr,
                     str filename,
                     mutable char c,
                     mutable char n,
                     mutable uint mark_line,
                     mutable uint mark_col,
                     mutable uint line,
                     mutable uint col,
                     hashmap[str,token.token] keywords,
                     hashmap[str,()] reserved) {

            fn is_eof() -> bool {
                ret c == (-1) as char;
            }

            fn get_curr_pos() -> common.pos {
                ret rec(line=line, col=col);
            }

            fn get_mark_pos() -> common.pos {
                ret rec(line=mark_line, col=mark_col);
            }

            fn get_filename() -> str {
                ret filename;
            }

            fn curr() -> char {
                ret c;
            }

            fn next() -> char {
                ret n;
            }

            impure fn bump() {

                let char prev = c;

                c = n;

                if (c == (-1) as char) {
                    ret;
                }

                if (prev == '\n') {
                    line += 1u;
                    col = 0u;
                } else {
                    col += 1u;
                }

                n = rdr.getc() as char;
            }

            fn mark() {
                mark_line = line;
                mark_col = col;
            }

            fn get_keywords() -> hashmap[str,token.token] {
                ret keywords;
            }

            fn get_reserved() -> hashmap[str,()] {
                ret reserved;
            }
        }

    auto keywords = new_str_hash[token.token]();
    auto reserved = new_str_hash[()]();

    keywords.insert("mod", token.MOD);
    keywords.insert("use", token.USE);
    keywords.insert("meta", token.META);
    keywords.insert("auth", token.AUTH);

    keywords.insert("syntax", token.SYNTAX);

    keywords.insert("if", token.IF);
    keywords.insert("else", token.ELSE);
    keywords.insert("while", token.WHILE);
    keywords.insert("do", token.DO);
    keywords.insert("alt", token.ALT);
    keywords.insert("case", token.CASE);

    keywords.insert("for", token.FOR);
    keywords.insert("each", token.EACH);
    keywords.insert("put", token.PUT);
    keywords.insert("ret", token.RET);
    keywords.insert("be", token.BE);

    keywords.insert("fail", token.FAIL);
    keywords.insert("drop", token.DROP);

    keywords.insert("type", token.TYPE);
    keywords.insert("check", token.CHECK);
    keywords.insert("claim", token.CLAIM);
    keywords.insert("prove", token.PROVE);

    keywords.insert("abs", token.ABS);

    keywords.insert("state", token.STATE);
    keywords.insert("gc", token.GC);

    keywords.insert("impure", token.IMPURE);
    keywords.insert("unsafe", token.UNSAFE);

    keywords.insert("native", token.NATIVE);
    keywords.insert("mutable", token.MUTABLE);
    keywords.insert("auto", token.AUTO);

    keywords.insert("fn", token.FN);
    keywords.insert("iter", token.ITER);

    keywords.insert("import", token.IMPORT);
    keywords.insert("export", token.EXPORT);

    keywords.insert("let", token.LET);
    keywords.insert("const", token.CONST);

    keywords.insert("log", token.LOG);
    keywords.insert("spawn", token.SPAWN);
    keywords.insert("thread", token.THREAD);
    keywords.insert("yield", token.YIELD);
    keywords.insert("join", token.JOIN);

    keywords.insert("bool", token.BOOL);

    keywords.insert("int", token.INT);
    keywords.insert("uint", token.UINT);
    keywords.insert("float", token.FLOAT);

    keywords.insert("char", token.CHAR);
    keywords.insert("str", token.STR);


    keywords.insert("rec", token.REC);
    keywords.insert("tup", token.TUP);
    keywords.insert("tag", token.TAG);
    keywords.insert("vec", token.VEC);
    keywords.insert("any", token.ANY);

    keywords.insert("obj", token.OBJ);

    keywords.insert("port", token.PORT);
    keywords.insert("chan", token.CHAN);

    keywords.insert("task", token.TASK);

    keywords.insert("true", token.LIT_BOOL(true));
    keywords.insert("false", token.LIT_BOOL(false));

    keywords.insert("in", token.IN);

    keywords.insert("as", token.AS);
    keywords.insert("with", token.WITH);

    keywords.insert("bind", token.BIND);

    keywords.insert("u8", token.MACH(common.ty_u8));
    keywords.insert("u16", token.MACH(common.ty_u16));
    keywords.insert("u32", token.MACH(common.ty_u32));
    keywords.insert("u64", token.MACH(common.ty_u64));
    keywords.insert("i8", token.MACH(common.ty_i8));
    keywords.insert("i16", token.MACH(common.ty_i16));
    keywords.insert("i32", token.MACH(common.ty_i32));
    keywords.insert("i64", token.MACH(common.ty_i64));
    keywords.insert("f32", token.MACH(common.ty_f32));
    keywords.insert("f64", token.MACH(common.ty_f64));

    ret reader(rdr, filename, rdr.getc() as char, rdr.getc() as char,
               1u, 0u, 1u, 0u, keywords, reserved);
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

impure fn consume_any_whitespace(reader rdr) {
    while (is_whitespace(rdr.curr())) {
        rdr.bump();
    }
    be consume_any_line_comment(rdr);
}

impure fn consume_any_line_comment(reader rdr) {
    if (rdr.curr() == '/') {
        alt (rdr.next()) {
            case ('/') {
                while (rdr.curr() != '\n') {
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


impure fn consume_block_comment(reader rdr) {
    let int level = 1;
    while (level > 0) {
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

impure fn scan_number(mutable char c, reader rdr) -> token.token {
    auto accum_int = 0;
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
    }

    if (c == '0' && n == 'b') {
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
    }

    while (is_dec_digit(c) || c == '_') {
        if (c != '_') {
            accum_int *= 10;
            accum_int += dec_digit_val(c);
        }
        rdr.bump();
        c = rdr.curr();
    }

    if (c == 'u' || c == 'i') {
        let bool signed = (c == 'i');
        rdr.bump();
        c = rdr.curr();
        if (c == '8') {
            rdr.bump();
            if (signed) {
                ret token.LIT_MACH_INT(common.ty_i8, accum_int);
            } else {
                ret token.LIT_MACH_INT(common.ty_u8, accum_int);
            }
        }

        n = rdr.next();
        if (c == '1' && n == '6') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token.LIT_MACH_INT(common.ty_i16, accum_int);
            } else {
                ret token.LIT_MACH_INT(common.ty_u16, accum_int);
            }
        }
        if (c == '3' && n == '2') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token.LIT_MACH_INT(common.ty_i32, accum_int);
            } else {
                ret token.LIT_MACH_INT(common.ty_u32, accum_int);
            }
        }

        if (c == '6' && n == '4') {
            rdr.bump();
            rdr.bump();
            if (signed) {
                ret token.LIT_MACH_INT(common.ty_i64, accum_int);
            } else {
                ret token.LIT_MACH_INT(common.ty_u64, accum_int);
            }
        }

        if (signed) {
            ret token.LIT_INT(accum_int);
        } else {
            // FIXME: should cast in the target bit-width.
            ret token.LIT_UINT(accum_int as uint);
        }
    }
    ret token.LIT_INT(accum_int);
}

impure fn next_token(reader rdr) -> token.token {
    auto accum_str = "";

    consume_any_whitespace(rdr);

    if (rdr.is_eof()) { ret token.EOF; }

    rdr.mark();
    auto c = rdr.curr();

    if (is_alpha(c) || c == '_') {
        while (is_alnum(c) || c == '_') {
            accum_str += (c as u8);
            rdr.bump();
            c = rdr.curr();
        }

        if (_str.eq(accum_str, "_")) {
            ret token.UNDERSCORE;
        }

        auto kwds = rdr.get_keywords();
        if (kwds.contains_key(accum_str)) {
            ret kwds.get(accum_str);
        }

        ret token.IDENT(accum_str);
    }

    if (is_dec_digit(c)) {
        ret scan_number(c, rdr);
    }

    impure fn binop(reader rdr, token.binop op) -> token.token {
        rdr.bump();
        if (rdr.curr() == '=') {
            rdr.bump();
            ret token.BINOPEQ(op);
        } else {
            ret token.BINOP(op);
        }
    }

    alt (c) {
        // One-byte tokens.
        case (':') { rdr.bump(); ret token.COLON; }
        case ('?') { rdr.bump(); ret token.QUES; }
        case (';') { rdr.bump(); ret token.SEMI; }
        case (',') { rdr.bump(); ret token.COMMA; }
        case ('.') { rdr.bump(); ret token.DOT; }
        case ('(') { rdr.bump(); ret token.LPAREN; }
        case (')') { rdr.bump(); ret token.RPAREN; }
        case ('{') { rdr.bump(); ret token.LBRACE; }
        case ('}') { rdr.bump(); ret token.RBRACE; }
        case ('[') { rdr.bump(); ret token.LBRACKET; }
        case (']') { rdr.bump(); ret token.RBRACKET; }
        case ('@') { rdr.bump(); ret token.AT; }
        case ('#') { rdr.bump(); ret token.POUND; }
        case ('~') { rdr.bump(); ret token.TILDE; }


        // Multi-byte tokens.
        case ('=') {
            rdr.bump();
            if (rdr.curr() == '=') {
                rdr.bump();
                ret token.EQEQ;
            } else {
                ret token.EQ;
            }
        }

        case ('!') {
            rdr.bump();
            if (rdr.curr() == '=') {
                rdr.bump();
                ret token.NE;
            } else {
                ret token.NOT;
            }
        }

        case ('<') {
            rdr.bump();
            alt (rdr.curr()) {
                case ('=') {
                    rdr.bump();
                    ret token.LE;
                }
                case ('<') {
                    ret binop(rdr, token.LSL);
                }
                case ('-') {
                    rdr.bump();
                    ret token.LARROW;
                }
                case ('|') {
                    rdr.bump();
                    ret token.SEND;
                }
                case (_) {
                    ret token.LT;
                }
            }
        }

        case ('>') {
            rdr.bump();
            alt (rdr.curr()) {
                case ('=') {
                    rdr.bump();
                    ret token.GE;
                }

                case ('>') {
                    if (rdr.next() == '>') {
                        rdr.bump();
                        ret binop(rdr, token.ASR);
                    } else {
                        ret binop(rdr, token.LSR);
                    }
                }

                case (_) {
                    ret token.GT;
                }
            }
        }

        case ('\'') {
            rdr.bump();
            auto c2 = rdr.curr();
            if (c2 == '\\') {
                alt (rdr.next()) {
                    case ('n') { rdr.bump(); c2 = '\n'; }
                    case ('r') { rdr.bump(); c2 = '\r'; }
                    case ('t') { rdr.bump(); c2 = '\t'; }
                    case ('\\') { rdr.bump(); c2 = '\\'; }
                    case ('\'') { rdr.bump(); c2 = '\''; }
                    // FIXME: unicode numeric escapes.
                    case (?c2) {
                        log "unknown character escape";
                        log c2;
                        fail;
                    }
                }
            }

            if (rdr.next() != '\'') {
                log "unterminated character constant";
                fail;
            }
            rdr.bump();
            rdr.bump();
            ret token.LIT_CHAR(c2);
        }

        case ('"') {
            rdr.bump();
            // FIXME: general utf8-consumption support.
            while (rdr.curr() != '"') {
                alt (rdr.curr()) {
                    case ('\\') {
                        alt (rdr.next()) {
                            case ('n') {
                                rdr.bump();
                                accum_str += '\n' as u8;
                            }
                            case ('r') {
                                rdr.bump();
                                accum_str += '\r' as u8;
                            }
                            case ('t') {
                                rdr.bump();
                                accum_str += '\t' as u8;
                            }
                            case ('\\') {
                                rdr.bump();
                                accum_str += '\\' as u8;
                            }
                            case ('"') {
                                rdr.bump();
                                accum_str += '"' as u8;
                            }
                            // FIXME: unicode numeric escapes.
                            case (?c2) {
                                log "unknown string escape";
                                log c2;
                                fail;
                            }
                        }
                    }
                    case (_) {
                        accum_str += rdr.curr() as u8;
                    }
                }
                rdr.bump();
            }
            rdr.bump();
            ret token.LIT_STR(accum_str);
        }

        case ('-') {
            if (rdr.next() == '>') {
                rdr.bump();
                rdr.bump();
                ret token.RARROW;
            } else {
                ret binop(rdr, token.MINUS);
            }
        }

        case ('&') {
            if (rdr.next() == '&') {
                rdr.bump();
                rdr.bump();
                ret token.ANDAND;
            } else {
                ret binop(rdr, token.AND);
            }
        }

        case ('|') {
            if (rdr.next() == '|') {
                rdr.bump();
                rdr.bump();
                ret token.OROR;
            } else {
                ret binop(rdr, token.OR);
            }
        }

        case ('+') {
            ret binop(rdr, token.PLUS);
        }

        case ('*') {
            ret binop(rdr, token.STAR);
        }

        case ('/') {
            ret binop(rdr, token.SLASH);
        }

        case ('^') {
            ret binop(rdr, token.CARET);
        }

        case ('%') {
            ret binop(rdr, token.PERCENT);
        }

    }

    log "lexer stopping at ";
    log c;
    ret token.EOF;
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

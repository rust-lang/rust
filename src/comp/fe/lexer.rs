import std._io.stdio_reader;
import std._str;
import std.map;
import std.map.hashmap;

fn new_str_hash[V]() -> map.hashmap[str,V] {
    let map.hashfn[str] hasher = _str.hash;
    let map.eqfn[str] eqer = _str.eq;
    ret map.mk_hashmap[str,V](hasher, eqer);
}

type reader = obj {
              fn is_eof() -> bool;
              fn peek() -> char;
              fn bump();
              fn get_pos() -> tup(str,uint,uint);
              fn get_keywords() -> hashmap[str,token.token];
              fn get_reserved() -> hashmap[str,()];
};

fn new_reader(stdio_reader rdr, str filename) -> reader
{
    obj reader(stdio_reader rdr,
               str filename,
               mutable char c,
               mutable uint line,
               mutable uint col,
               hashmap[str,token.token] keywords,
               hashmap[str,()] reserved)
        {
            fn is_eof() -> bool {
                ret c == (-1) as char;
            }

            fn get_pos() -> tup(str,uint,uint) {
                ret tup(filename, line, col);
            }

            fn peek() -> char {
                ret c;
            }

            fn bump() {
                c = rdr.getc() as char;
                if (c == '\n') {
                    line += 1u;
                    col = 0u;
                } else {
                    col += 1u;
                }
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

    keywords.insert("mod", token.MOD());
    keywords.insert("use", token.USE());
    keywords.insert("meta", token.META());
    keywords.insert("auth", token.AUTH());

    keywords.insert("syntax", token.SYNTAX());

    keywords.insert("if", token.IF());
    keywords.insert("else", token.ELSE());
    keywords.insert("while", token.WHILE());
    keywords.insert("do", token.DO());
    keywords.insert("alt", token.ALT());
    keywords.insert("case", token.CASE());

    keywords.insert("for", token.FOR());
    keywords.insert("each", token.EACH());
    keywords.insert("put", token.PUT());
    keywords.insert("ret", token.RET());
    keywords.insert("be", token.BE());

    ret reader(rdr, filename, rdr.getc() as char, 1u, 1u,
               keywords, reserved);
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

fn is_hex_digit(char c) -> bool {
    ret in_range(c, '0', '9') ||
        in_range(c, 'a', 'f') ||
        in_range(c, 'A', 'F');
}

fn is_bin_digit(char c) -> bool {
    ret c == '0' || c == '1';
}

fn is_whitespace(char c) -> bool {
    ret c == ' ' || c == '\t' || c == '\r' || c == '\n';
}

fn consume_any_whitespace(stdio_reader rdr, char c) -> char {
    auto c1 = c;
    while (is_whitespace(c1)) {
        c1 = rdr.getc() as char;
    }
    be consume_any_line_comment(rdr, c1);
}

fn consume_any_line_comment(stdio_reader rdr, char c) -> char {
    auto c1 = c;
    if (c1 == '/') {
        auto c2 = rdr.getc() as char;
        if (c2 == '/') {
            while (c1 != '\n') {
                c1 = rdr.getc() as char;
            }
            // Restart whitespace munch.
            be consume_any_whitespace(rdr, c1);
        }
    }
    ret c;
}

fn next_token(stdio_reader rdr) -> token.token {
    auto eof = (-1) as char;
    auto c = rdr.getc() as char;
    auto accum_str = "";
    auto accum_int = 0;

    fn next(stdio_reader rdr) -> char {
        ret rdr.getc() as char;
    }

    fn forget(stdio_reader rdr, char c) {
        rdr.ungetc(c as int);
    }

    c = consume_any_whitespace(rdr, c);

    if (c == eof) { ret token.EOF(); }

    if (is_alpha(c)) {
        while (is_alpha(c)) {
            accum_str += (c as u8);
            c = next(rdr);
        }
        forget(rdr, c);
        ret token.IDENT(accum_str);
    }

    if (is_dec_digit(c)) {
        if (c == '0') {
        } else {
            while (is_dec_digit(c)) {
                accum_int *= 10;
                accum_int += (c as int) - ('0' as int);
                c = next(rdr);
            }
            forget(rdr, c);
            ret token.LIT_INT(accum_int);
        }
    }


    fn op_or_opeq(stdio_reader rdr, char c2,
                  token.op op) -> token.token {
        if (c2 == '=') {
            ret token.OPEQ(op);
        } else {
            forget(rdr, c2);
            ret token.OP(op);
        }
    }

    alt (c) {
        // One-byte tokens.
        case (';') { ret token.SEMI(); }
        case (',') { ret token.COMMA(); }
        case ('.') { ret token.DOT(); }
        case ('(') { ret token.LPAREN(); }
        case (')') { ret token.RPAREN(); }
        case ('{') { ret token.LBRACE(); }
        case ('}') { ret token.RBRACE(); }
        case ('[') { ret token.LBRACKET(); }
        case (']') { ret token.RBRACKET(); }
        case ('@') { ret token.AT(); }
        case ('#') { ret token.POUND(); }

        // Multi-byte tokens.
        case ('=') {
            auto c2 = next(rdr);
            if (c2 == '=') {
                ret token.OP(token.EQEQ());
            } else {
                forget(rdr, c2);
                ret token.OP(token.EQ());
            }
        }

        case ('\'') {
            // FIXME: general utf8-consumption support.
            auto c2 = next(rdr);
            if (c2 == '\\') {
                c2 = next(rdr);
                alt (c2) {
                    case ('n') { c2 = '\n'; }
                    case ('r') { c2 = '\r'; }
                    case ('t') { c2 = '\t'; }
                    case ('\\') { c2 = '\\'; }
                    case ('\'') { c2 = '\''; }
                    // FIXME: unicode numeric escapes.
                    case (_) {
                        log "unknown character escape";
                        log c2;
                        fail;
                    }
                }
            }
            if (next(rdr) != '\'') {
                log "unterminated character constant";
                fail;
            }
            ret token.LIT_CHAR(c2);
        }

        case ('"') {
            // FIXME: general utf8-consumption support.
            auto c2 = next(rdr);
            while (c2 != '"') {
                alt (c2) {
                    case ('\\') {
                        c2 = next(rdr);
                        alt (c2) {
                            case ('n') { accum_str += '\n' as u8; }
                            case ('r') { accum_str += '\r' as u8; }
                            case ('t') { accum_str += '\t' as u8; }
                            case ('\\') { accum_str += '\\' as u8; }
                            case ('"') { accum_str += '"' as u8; }
                            // FIXME: unicode numeric escapes.
                            case (_) {
                                log "unknown string escape";
                                log c2;
                                fail;
                            }
                        }
                    }
                    case (_) {
                        accum_str += c2 as u8;
                    }
                }
                c2 = next(rdr);
            }
            ret token.LIT_STR(accum_str);
        }

        case ('-') {
            auto c2 = next(rdr);
            if (c2 == '>') {
                ret token.RARROW();
            } else {
                ret op_or_opeq(rdr, c2, token.MINUS());
            }
        }

        case ('&') {
            auto c2 = next(rdr);
            if (c2 == '&') {
                ret token.OP(token.ANDAND());
            } else {
                ret op_or_opeq(rdr, c2, token.AND());
            }
        }

        case ('+') {
            ret op_or_opeq(rdr, next(rdr), token.PLUS());
        }

        case ('*') {
            ret op_or_opeq(rdr, next(rdr), token.STAR());
        }

        case ('/') {
            ret op_or_opeq(rdr, next(rdr), token.STAR());
        }

        case ('!') {
            ret op_or_opeq(rdr, next(rdr), token.NOT());
        }

        case ('^') {
            ret op_or_opeq(rdr, next(rdr), token.CARET());
        }

        case ('%') {
            ret op_or_opeq(rdr, next(rdr), token.PERCENT());
        }

    }

    log "lexer stopping at ";
    log c;
    ret token.EOF();
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

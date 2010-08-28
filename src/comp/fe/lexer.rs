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
              fn curr() -> char;
              fn next() -> char;
              fn bump();
              fn get_curr_pos() -> tup(str,uint,uint);
              fn get_keywords() -> hashmap[str,token.token];
              fn get_reserved() -> hashmap[str,()];
};

fn new_reader(stdio_reader rdr, str filename) -> reader
{
    obj reader(stdio_reader rdr,
               str filename,
               mutable char c,
               mutable char n,
               mutable uint line,
               mutable uint col,
               hashmap[str,token.token] keywords,
               hashmap[str,()] reserved)
        {
            fn is_eof() -> bool {
                ret c == (-1) as char;
            }

            fn get_curr_pos() -> tup(str,uint,uint) {
                ret tup(filename, line, col);
            }

            fn curr() -> char {
                ret c;
            }

            fn next() -> char {
                ret n;
            }

            fn bump() {
                c = n;

                if (c == (-1) as char) {
                    ret;
                }

                if (c == '\n') {
                    line += 1u;
                    col = 0u;
                } else {
                    col += 1u;
                }

                n = rdr.getc() as char;
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

    ret reader(rdr, filename, rdr.getc() as char, rdr.getc() as char,
               1u, 1u, keywords, reserved);
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

fn consume_any_whitespace(reader rdr) {
    while (is_whitespace(rdr.curr())) {
        rdr.bump();
    }
    be consume_any_line_comment(rdr);
}

fn consume_any_line_comment(reader rdr) {
    if (rdr.curr() == '/') {
        if (rdr.next() == '/') {
            while (rdr.curr() != '\n') {
                rdr.bump();
            }
            // Restart whitespace munch.
            be consume_any_whitespace(rdr);
        }
    }
}

fn next_token(reader rdr) -> token.token {
    auto accum_str = "";
    auto accum_int = 0;

    consume_any_whitespace(rdr);

    if (rdr.is_eof()) { ret token.EOF(); }

    auto c = rdr.curr();

    if (is_alpha(c)) {
        while (is_alpha(rdr.curr())) {
            c = rdr.curr();
            accum_str += (c as u8);
            rdr.bump();
        }
        ret token.IDENT(accum_str);
    }

    if (is_dec_digit(c)) {
        if (c == '0') {
            log "fixme: leading zero";
            fail;
        } else {
            while (is_dec_digit(c)) {
                c = rdr.curr();
                accum_int *= 10;
                accum_int += (c as int) - ('0' as int);
                rdr.bump();
            }
            ret token.LIT_INT(accum_int);
        }
    }


    fn op_or_opeq(reader rdr, token.op op) -> token.token {
        rdr.bump();
        if (rdr.next() == '=') {
            rdr.bump();
            ret token.OPEQ(op);
        } else {
            ret token.OP(op);
        }
    }

    alt (c) {
        // One-byte tokens.
        case (';') { rdr.bump(); ret token.SEMI(); }
        case (',') { rdr.bump(); ret token.COMMA(); }
        case ('.') { rdr.bump(); ret token.DOT(); }
        case ('(') { rdr.bump(); ret token.LPAREN(); }
        case (')') { rdr.bump(); ret token.RPAREN(); }
        case ('{') { rdr.bump(); ret token.LBRACE(); }
        case ('}') { rdr.bump(); ret token.RBRACE(); }
        case ('[') { rdr.bump(); ret token.LBRACKET(); }
        case (']') { rdr.bump(); ret token.RBRACKET(); }
        case ('@') { rdr.bump(); ret token.AT(); }
        case ('#') { rdr.bump(); ret token.POUND(); }

        // Multi-byte tokens.
        case ('=') {
            if (rdr.next() == '=') {
                rdr.bump();
                rdr.bump();
                ret token.OP(token.EQEQ());
            } else {
                rdr.bump();
                ret token.OP(token.EQ());
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
                    case (c2) {
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
                            case (c2) {
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
                ret token.RARROW();
            } else {
                ret op_or_opeq(rdr, token.MINUS());
            }
        }

        case ('&') {
            if (rdr.next() == '&') {
                rdr.bump();
                rdr.bump();
                ret token.OP(token.ANDAND());
            } else {
                ret op_or_opeq(rdr, token.AND());
            }
        }

        case ('+') {
            ret op_or_opeq(rdr, token.PLUS());
        }

        case ('*') {
            ret op_or_opeq(rdr, token.STAR());
        }

        case ('/') {
            ret op_or_opeq(rdr, token.STAR());
        }

        case ('!') {
            ret op_or_opeq(rdr, token.NOT());
        }

        case ('^') {
            ret op_or_opeq(rdr, token.CARET());
        }

        case ('%') {
            ret op_or_opeq(rdr, token.PERCENT());
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

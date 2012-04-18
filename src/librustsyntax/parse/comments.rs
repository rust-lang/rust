import io::reader_util;
import util::interner;
import lexer::{ reader, new_reader, next_token, is_whitespace };

export cmnt;
export lit;
export cmnt_style;
export gather_comments_and_literals;

enum cmnt_style {
    isolated, // No code on either side of each line of the comment
    trailing, // Code exists to the left of the comment
    mixed, // Code before /* foo */ and after the comment
    blank_line, // Just a manual blank line "\n\n", for layout
}

type cmnt = {style: cmnt_style, lines: [str], pos: uint};

fn read_to_eol(rdr: reader) -> str {
    let mut val = "";
    while rdr.curr != '\n' && !rdr.is_eof() {
        str::push_char(val, rdr.curr);
        rdr.bump();
    }
    if rdr.curr == '\n' { rdr.bump(); }
    ret val;
}

fn read_one_line_comment(rdr: reader) -> str {
    let val = read_to_eol(rdr);
    assert (val[0] == '/' as u8 && val[1] == '/' as u8);
    ret val;
}

fn consume_non_eol_whitespace(rdr: reader) {
    while is_whitespace(rdr.curr) && rdr.curr != '\n' && !rdr.is_eof() {
        rdr.bump();
    }
}

fn push_blank_line_comment(rdr: reader, &comments: [cmnt]) {
    #debug(">>> blank-line comment");
    let v: [str] = [];
    comments += [{style: blank_line, lines: v, pos: rdr.chpos}];
}

fn consume_whitespace_counting_blank_lines(rdr: reader, &comments: [cmnt]) {
    while is_whitespace(rdr.curr) && !rdr.is_eof() {
        if rdr.col == 0u && rdr.curr == '\n' {
            push_blank_line_comment(rdr, comments);
        }
        rdr.bump();
    }
}

fn read_line_comments(rdr: reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> line comments");
    let p = rdr.chpos;
    let mut lines: [str] = [];
    while rdr.curr == '/' && rdr.next() == '/' {
        let line = read_one_line_comment(rdr);
        log(debug, line);
        lines += [line];
        consume_non_eol_whitespace(rdr);
    }
    #debug("<<< line comments");
    ret {style: if code_to_the_left { trailing } else { isolated },
         lines: lines,
         pos: p};
}

fn all_whitespace(s: str, begin: uint, end: uint) -> bool {
    let mut i: uint = begin;
    while i != end { if !is_whitespace(s[i] as char) { ret false; } i += 1u; }
    ret true;
}

fn trim_whitespace_prefix_and_push_line(&lines: [str],
                                        s: str, col: uint) unsafe {
    let mut s1;
    if all_whitespace(s, 0u, col) {
        if col < str::len(s) {
            s1 = str::slice(s, col, str::len(s));
        } else { s1 = ""; }
    } else { s1 = s; }
    log(debug, "pushing line: " + s1);
    lines += [s1];
}

fn read_block_comment(rdr: reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> block comment");
    let p = rdr.chpos;
    let mut lines: [str] = [];
    let mut col: uint = rdr.col;
    rdr.bump();
    rdr.bump();
    let mut curr_line = "/*";
    let mut level: int = 1;
    while level > 0 {
        #debug("=== block comment level %d", level);
        if rdr.is_eof() { rdr.fatal("unterminated block comment"); }
        if rdr.curr == '\n' {
            trim_whitespace_prefix_and_push_line(lines, curr_line, col);
            curr_line = "";
            rdr.bump();
        } else {
            str::push_char(curr_line, rdr.curr);
            if rdr.curr == '/' && rdr.next() == '*' {
                rdr.bump();
                rdr.bump();
                curr_line += "*";
                level += 1;
            } else {
                if rdr.curr == '*' && rdr.next() == '/' {
                    rdr.bump();
                    rdr.bump();
                    curr_line += "/";
                    level -= 1;
                } else { rdr.bump(); }
            }
        }
    }
    if str::len(curr_line) != 0u {
        trim_whitespace_prefix_and_push_line(lines, curr_line, col);
    }
    let mut style = if code_to_the_left { trailing } else { isolated };
    consume_non_eol_whitespace(rdr);
    if !rdr.is_eof() && rdr.curr != '\n' && vec::len(lines) == 1u {
        style = mixed;
    }
    #debug("<<< block comment");
    ret {style: style, lines: lines, pos: p};
}

fn peeking_at_comment(rdr: reader) -> bool {
    ret rdr.curr == '/' && rdr.next() == '/' ||
            rdr.curr == '/' && rdr.next() == '*';
}

fn consume_comment(rdr: reader, code_to_the_left: bool, &comments: [cmnt]) {
    #debug(">>> consume comment");
    if rdr.curr == '/' && rdr.next() == '/' {
        comments += [read_line_comments(rdr, code_to_the_left)];
    } else if rdr.curr == '/' && rdr.next() == '*' {
        comments += [read_block_comment(rdr, code_to_the_left)];
    } else { fail; }
    #debug("<<< consume comment");
}

fn is_lit(t: token::token) -> bool {
    ret alt t {
          token::LIT_INT(_, _) { true }
          token::LIT_UINT(_, _) { true }
          token::LIT_FLOAT(_, _) { true }
          token::LIT_STR(_) { true }
          token::LIT_BOOL(_) { true }
          _ { false }
        }
}

type lit = {lit: str, pos: uint};

fn gather_comments_and_literals(span_diagnostic: diagnostic::span_handler,
                                path: str,
                                srdr: io::reader) ->
   {cmnts: [cmnt], lits: [lit]} {
    let src = @str::from_bytes(srdr.read_whole_stream());
    let itr = @interner::mk::<str>(str::hash, str::eq);
    let rdr = new_reader(span_diagnostic,
                         codemap::new_filemap(path, src, 0u, 0u), itr);
    let mut comments: [cmnt] = [];
    let mut literals: [lit] = [];
    let mut first_read: bool = true;
    while !rdr.is_eof() {
        loop {
            let mut code_to_the_left = !first_read;
            consume_non_eol_whitespace(rdr);
            if rdr.curr == '\n' {
                code_to_the_left = false;
                consume_whitespace_counting_blank_lines(rdr, comments);
            }
            while peeking_at_comment(rdr) {
                consume_comment(rdr, code_to_the_left, comments);
                consume_whitespace_counting_blank_lines(rdr, comments);
            }
            break;
        }
        let tok = next_token(rdr);
        if is_lit(tok.tok) {
            let s = rdr.get_str_from(tok.bpos);
            literals += [{lit: s, pos: tok.chpos}];
            log(debug, "tok lit: " + s);
        } else {
            log(debug, "tok: " + token::to_str(*rdr.interner, tok.tok));
        }
        first_read = false;
    }
    ret {cmnts: comments, lits: literals};
}

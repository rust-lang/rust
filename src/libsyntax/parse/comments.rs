import io::reader_util;
import io::println;//XXXXXXXXxxx
import util::interner;
import lexer::{ string_reader, bump, is_eof, nextch,
               is_whitespace, get_str_from, string_reader_as_reader };

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

type cmnt = {style: cmnt_style, lines: [str]/~, pos: uint};

fn read_to_eol(rdr: string_reader) -> str {
    let mut val = "";
    while rdr.curr != '\n' && !is_eof(rdr) {
        str::push_char(val, rdr.curr);
        bump(rdr);
    }
    if rdr.curr == '\n' { bump(rdr); }
    ret val;
}

fn read_one_line_comment(rdr: string_reader) -> str {
    let val = read_to_eol(rdr);
    assert ((val[0] == '/' as u8 && val[1] == '/' as u8) ||
            (val[0] == '#' as u8 && val[1] == '!' as u8));
    ret val;
}

fn consume_non_eol_whitespace(rdr: string_reader) {
    while is_whitespace(rdr.curr) && rdr.curr != '\n' && !is_eof(rdr) {
        bump(rdr);
    }
}

fn push_blank_line_comment(rdr: string_reader, &comments: [cmnt]/~) {
    #debug(">>> blank-line comment");
    let v: [str]/~ = []/~;
    comments += [{style: blank_line, lines: v, pos: rdr.chpos}]/~;
}

fn consume_whitespace_counting_blank_lines(rdr: string_reader,
                                           &comments: [cmnt]/~) {
    while is_whitespace(rdr.curr) && !is_eof(rdr) {
        if rdr.col == 0u && rdr.curr == '\n' {
            push_blank_line_comment(rdr, comments);
        }
        bump(rdr);
    }
}

fn read_shebang_comment(rdr: string_reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> shebang comment");
    let p = rdr.chpos;
    #debug("<<< shebang comment");
    ret {style: if code_to_the_left { trailing } else { isolated },
         lines: [read_one_line_comment(rdr)]/~,
         pos: p};
}

fn read_line_comments(rdr: string_reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> line comments");
    let p = rdr.chpos;
    let mut lines: [str]/~ = []/~;
    while rdr.curr == '/' && nextch(rdr) == '/' {
        let line = read_one_line_comment(rdr);
        log(debug, line);
        lines += [line]/~;
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

fn trim_whitespace_prefix_and_push_line(&lines: [str]/~,
                                        s: str, col: uint) unsafe {
    let mut s1;
    let len = str::len(s);
    if all_whitespace(s, 0u, uint::min(len, col)) {
        if col < len {
            s1 = str::slice(s, col, len);
        } else { s1 = ""; }
    } else { s1 = s; }
    log(debug, "pushing line: " + s1);
    lines += [s1]/~;
}

fn read_block_comment(rdr: string_reader, code_to_the_left: bool) -> cmnt {
    #debug(">>> block comment");
    let p = rdr.chpos;
    let mut lines: [str]/~ = []/~;
    let mut col: uint = rdr.col;
    bump(rdr);
    bump(rdr);
    let mut curr_line = "/*";
    let mut level: int = 1;
    while level > 0 {
        #debug("=== block comment level %d", level);
        if is_eof(rdr) {(rdr as reader).fatal("unterminated block comment");}
        if rdr.curr == '\n' {
            trim_whitespace_prefix_and_push_line(lines, curr_line, col);
            curr_line = "";
            bump(rdr);
        } else {
            str::push_char(curr_line, rdr.curr);
            if rdr.curr == '/' && nextch(rdr) == '*' {
                bump(rdr);
                bump(rdr);
                curr_line += "*";
                level += 1;
            } else {
                if rdr.curr == '*' && nextch(rdr) == '/' {
                    bump(rdr);
                    bump(rdr);
                    curr_line += "/";
                    level -= 1;
                } else { bump(rdr); }
            }
        }
    }
    if str::len(curr_line) != 0u {
        trim_whitespace_prefix_and_push_line(lines, curr_line, col);
    }
    let mut style = if code_to_the_left { trailing } else { isolated };
    consume_non_eol_whitespace(rdr);
    if !is_eof(rdr) && rdr.curr != '\n' && vec::len(lines) == 1u {
        style = mixed;
    }
    #debug("<<< block comment");
    ret {style: style, lines: lines, pos: p};
}

fn peeking_at_comment(rdr: string_reader) -> bool {
    ret ((rdr.curr == '/' && nextch(rdr) == '/') ||
         (rdr.curr == '/' && nextch(rdr) == '*')) ||
         (rdr.curr == '#' && nextch(rdr) == '!');
}

fn consume_comment(rdr: string_reader, code_to_the_left: bool,
                   &comments: [cmnt]/~) {
    #debug(">>> consume comment");
    if rdr.curr == '/' && nextch(rdr) == '/' {
        comments += [read_line_comments(rdr, code_to_the_left)]/~;
    } else if rdr.curr == '/' && nextch(rdr) == '*' {
        comments += [read_block_comment(rdr, code_to_the_left)]/~;
    } else if rdr.curr == '#' && nextch(rdr) == '!' {
        comments += [read_shebang_comment(rdr, code_to_the_left)]/~;
    } else { fail; }
    #debug("<<< consume comment");
}

type lit = {lit: str, pos: uint};

fn gather_comments_and_literals(span_diagnostic: diagnostic::span_handler,
                                path: str,
                                srdr: io::reader) ->
   {cmnts: [cmnt]/~, lits: [lit]/~} {
    let src = @str::from_bytes(srdr.read_whole_stream());
    let itr = @interner::mk::<@str>(
        {|x|str::hash(*x)},
        {|x,y|str::eq(*x, *y)}
    );
    let rdr = lexer::new_low_level_string_reader
        (span_diagnostic, codemap::new_filemap(path, src, 0u, 0u), itr);

    let mut comments: [cmnt]/~ = []/~;
    let mut literals: [lit]/~ = []/~;
    let mut first_read: bool = true;
    while !is_eof(rdr) {
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


        let bstart = rdr.pos;
        rdr.next_token();
        //discard, and look ahead; we're working with internal state
        let {tok: tok, sp: sp} = rdr.peek();
        if token::is_lit(tok) {
            let s = get_str_from(rdr, bstart);
            vec::push(literals, {lit: s, pos: sp.lo});
            log(debug, "tok lit: " + s);
        } else {
            log(debug, "tok: " + token::to_str(*rdr.interner, tok));
        }
        first_read = false;
    }
    ret {cmnts: comments, lits: literals};
}

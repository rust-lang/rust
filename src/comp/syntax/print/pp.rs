
import std::{io, vec, str};

/*
 * This pretty-printer is a direct reimplementation of Philip Karlton's
 * Mesa pretty-printer, as described in appendix A of
 *
 *     STAN-CS-79-770: "Pretty Printing", by Derek C. Oppen.
 *     Stanford Department of Computer Science, 1979.
 *
 * The algorithm's aim is to break a stream into as few lines as possible
 * while respecting the indentation-consistency requirements of the enclosing
 * block, and avoiding breaking at silly places on block boundaries, for
 * example, between "x" and ")" in "x)".
 *
 * I am implementing this algorithm because it comes with 20 pages of
 * documentation explaining its theory, and because it addresses the set of
 * concerns I've seen other pretty-printers fall down on. Weirdly. Even though
 * it's 32 years old and not written in Haskell. What can I say?
 *
 * Despite some redundancies and quirks in the way it's implemented in that
 * paper, I've opted to keep the implementation here as similar as I can,
 * changing only what was blatantly wrong, a typo, or sufficiently
 * non-idiomatic rust that it really stuck out.
 *
 * In particular you'll see a certain amount of churn related to INTEGER vs.
 * CARDINAL in the Mesa implementation. Mesa apparently interconverts the two
 * somewhat readily? In any case, I've used uint for indices-in-buffers and
 * ints for character-sizes-and-indentation-offsets. This respects the need
 * for ints to "go negative" while carrying a pending-calculation balance, and
 * helps differentiate all the numbers flying around internally (slightly).
 *
 * I also inverted the indentation arithmetic used in the print stack, since
 * the Mesa implementation (somewhat randomly) stores the offset on the print
 * stack in terms of margin-col rather than col itself. I store col.
 *
 * I also implemented a small change in the STRING token, in that I store an
 * explicit length for the string. For most tokens this is just the length of
 * the accompanying string. But it's necessary to permit it to differ, for
 * encoding things that are supposed to "go on their own line" -- certain
 * classes of comment and blank-line -- where relying on adjacent
 * hardbreak-like BREAK tokens with long blankness indication doesn't actually
 * work. To see why, consider when there is a "thing that should be on its own
 * line" between two long blocks, say functions. If you put a hardbreak after
 * each function (or before each) and the breaking algorithm decides to break
 * there anyways (because the functions themselves are long) you wind up with
 * extra blank lines. If you don't put hardbreaks you can wind up with the
 * "thing which should be on its own line" not getting its own line in the
 * rare case of "really small functions" or such. This re-occurs with comments
 * and explicit blank lines. So in those cases we use a string with a payload
 * we want isolated to a line and an explicit length that's huge, surrounded
 * by two zero-length breaks. The algorithm will try its best to fit it on a
 * line (which it can't) and so naturally place the content on its own line to
 * avoid combining it with other lines and making matters even worse.
 */
tag breaks { consistent; inconsistent; }

type break_t = {offset: int, blank_space: int};

type begin_t = {offset: int, breaks: breaks};

tag token { STRING(str, int); BREAK(break_t); BEGIN(begin_t); END; EOF; }

fn tok_str(t: token) -> str {
    alt t {
      STRING(s, len) { ret #fmt["STR(%s,%d)", s, len]; }
      BREAK(_) { ret "BREAK"; }
      BEGIN(_) { ret "BEGIN"; }
      END. { ret "END"; }
      EOF. { ret "EOF"; }
    }
}

fn buf_str(toks: [mutable token], szs: [mutable int], left: uint, right: uint,
           lim: uint) -> str {
    let n = vec::len(toks);
    assert (n == vec::len(szs));
    let i = left;
    let L = lim;
    let s = "[";
    while i != right && L != 0u {
        L -= 1u;
        if i != left { s += ", "; }
        s += #fmt["%d=%s", szs[i], tok_str(toks[i])];
        i += 1u;
        i %= n;
    }
    s += "]";
    ret s;
}

tag print_stack_break { fits; broken(breaks); }

type print_stack_elt = {offset: int, pbreak: print_stack_break};

const size_infinity: int = 0xffff;

fn mk_printer(out: io::writer, linewidth: uint) -> printer {
    // Yes 3, it makes the ring buffers big enough to never
    // fall behind.

    let n: uint = 3u * linewidth;
    log #fmt["mk_printer %u", linewidth];
    let token: [mutable token] = vec::init_elt_mut(EOF, n);
    let size: [mutable int] = vec::init_elt_mut(0, n);
    let scan_stack: [mutable uint] = vec::init_elt_mut(0u, n);
    let print_stack: [print_stack_elt] = [];
    ret printer(out, n, linewidth as int, // margin
                linewidth as int, // space
                0u, // left
                0u, // right
                token, size, 0, // left_total
                0, // right_total
                scan_stack, true, // scan_stack_empty
                0u, // top
                0u, // bottom
                print_stack, 0);
}


/*
 * In case you do not have the paper, here is an explanation of what's going
 * on.
 *
 * There is a stream of input tokens flowing through this printer.
 *
 * The printer buffers up to 3N tokens inside itself, where N is linewidth.
 * Yes, linewidth is chars and tokens are multi-char, but in the worst
 * case every token worth buffering is 1 char long, so it's ok.
 *
 * Tokens are STRING, BREAK, and BEGIN/END to delimit blocks.
 *
 * BEGIN tokens can carry an offset, saying "how far to indent when you break
 * inside here", as well as a flag indicating "consistent" or "inconsistent"
 * breaking. Consistent breaking means that after the first break, no attempt
 * will be made to flow subsequent breaks together onto lines. Inconsistent
 * is the opposite. Inconsistent breaking example would be, say:
 *
 *  foo(hello, there, good, friends)
 *
 * breaking inconsistently to become
 *
 *  foo(hello, there
 *      good, friends);
 *
 * whereas a consistent breaking would yield:
 *
 *  foo(hello,
 *      there
 *      good,
 *      friends);
 *
 * That is, in the consistent-break blocks we value vertical alignment
 * more than the ability to cram stuff onto a line. But in all cases if it
 * can make a block a one-liner, it'll do so.
 *
 * Carrying on with high-level logic:
 *
 * The buffered tokens go through a ring-buffer, 'tokens'. The 'left' and
 * 'right' indices denote the active portion of the ring buffer as well as
 * describing hypothetical points-in-the-infinite-stream at most 3N tokens
 * apart (i.e. "not wrapped to ring-buffer boundaries"). The paper will switch
 * between using 'left' and 'right' terms to denote the wrapepd-to-ring-buffer
 * and point-in-infinite-stream senses freely.
 *
 * There is a parallel ring buffer, 'size', that holds the calculated size of
 * each token. Why calculated? Because for BEGIN/END pairs, the "size"
 * includes everything betwen the pair. That is, the "size" of BEGIN is
 * actually the sum of the sizes of everything between BEGIN and the paired
 * END that follows. Since that is arbitrarily far in the future, 'size' is
 * being rewritten regularly while the printer runs; in fact most of the
 * machinery is here to work out 'size' entries on the fly (and give up when
 * they're so obviously over-long that "infinity" is a good enough
 * approximation for purposes of line breaking).
 *
 * The "input side" of the printer is managed as an abstract process called
 * SCAN, which uses 'scan_stack', 'scan_stack_empty', 'top' and 'bottom', to
 * manage calculating 'size'. SCAN is, in other words, the process of
 * calculating 'size' entries.
 *
 * The "output side" of the printer is managed by an abstract process called
 * PRINT, which uses 'print_stack', 'margin' and 'space' to figure out what to
 * do with each token/size pair it consumes as it goes. It's trying to consume
 * the entire buffered window, but can't output anything until the size is >=
 * 0 (sizes are set to negative while they're pending calculation).
 *
 * So SCAN takeks input and buffers tokens and pending calculations, while
 * PRINT gobbles up completed calculations and tokens from the buffer. The
 * theory is that the two can never get more than 3N tokens apart, because
 * once there's "obviously" too much data to fit on a line, in a size
 * calculation, SCAN will write "infinity" to the size and let PRINT consume
 * it.
 *
 * In this implementation (following the paper, again) the SCAN process is
 * the method called 'pretty_print', and the 'PRINT' process is the method
 * called 'print'.
 */
obj printer(out: io::writer,
            buf_len: uint,
            mutable margin: int, // width of lines we're constrained to

            mutable space: int, // number of spaces left on line

            mutable left: uint, // index of left side of input stream

            mutable right: uint, // index of right side of input stream

            mutable token: [mutable token],

            // ring-buffr stream goes through
            mutable size: [mutable int], // ring-buffer of calculated sizes

            mutable left_total: int, // running size of stream "...left"

            mutable right_total: int, // running size of stream "...right"

             // pseudo-stack, really a ring too. Holds the
             // primary-ring-buffers index of the BEGIN that started the
             // current block, possibly with the most recent BREAK after that
             // BEGIN (if there is any) on top of it. Stuff is flushed off the
             // bottom as it becomes irrelevant due to the primary ring-buffer
             // advancing.
             mutable scan_stack: [mutable uint],
            mutable scan_stack_empty: bool, // top==bottom disambiguator

            mutable top: uint, // index of top of scan_stack

            mutable bottom: uint, // index of bottom of scan_stack

             // stack of blocks-in-progress being flushed by print
            mutable print_stack: [print_stack_elt],


            // buffered indentation to avoid writing trailing whitespace
            mutable pending_indentation: int) {

    fn last_token() -> token { ret token[right]; }

    // be very careful with this!
    fn replace_last_token(t: token) { token[right] = t; }

    fn pretty_print(t: token) {
        log #fmt["pp [%u,%u]", left, right];
        alt t {
          EOF. {
            if !scan_stack_empty {
                self.check_stack(0);
                self.advance_left(token[left], size[left]);
            }
            self.indent(0);
          }
          BEGIN(b) {
            if scan_stack_empty {
                left_total = 1;
                right_total = 1;
                left = 0u;
                right = 0u;
            } else { self.advance_right(); }
            log #fmt["pp BEGIN/buffer [%u,%u]", left, right];
            token[right] = t;
            size[right] = -right_total;
            self.scan_push(right);
          }
          END. {
            if scan_stack_empty {
                log #fmt["pp END/print [%u,%u]", left, right];
                self.print(t, 0);
            } else {
                log #fmt["pp END/buffer [%u,%u]", left, right];
                self.advance_right();
                token[right] = t;
                size[right] = -1;
                self.scan_push(right);
            }
          }
          BREAK(b) {
            if scan_stack_empty {
                left_total = 1;
                right_total = 1;
                left = 0u;
                right = 0u;
            } else { self.advance_right(); }
            log #fmt["pp BREAK/buffer [%u,%u]", left, right];
            self.check_stack(0);
            self.scan_push(right);
            token[right] = t;
            size[right] = -right_total;
            right_total += b.blank_space;
          }
          STRING(s, len) {
            if scan_stack_empty {
                log #fmt["pp STRING/print [%u,%u]", left, right];
                self.print(t, len);
            } else {
                log #fmt["pp STRING/buffer [%u,%u]", left, right];
                self.advance_right();
                token[right] = t;
                size[right] = len;
                right_total += len;
                self.check_stream();
            }
          }
        }
    }
    fn check_stream() {
        log #fmt["check_stream [%u, %u] with left_total=%d, right_total=%d",
                 left, right, left_total, right_total];
        if right_total - left_total > space {
            log #fmt["scan window is %d, longer than space on line (%d)",
                     right_total - left_total, space];
            if !scan_stack_empty {
                if left == scan_stack[bottom] {
                    log #fmt["setting %u to infinity and popping", left];
                    size[self.scan_pop_bottom()] = size_infinity;
                }
            }
            self.advance_left(token[left], size[left]);
            if left != right { self.check_stream(); }
        }
    }
    fn scan_push(x: uint) {
        log #fmt["scan_push %u", x];
        if scan_stack_empty {
            scan_stack_empty = false;
        } else { top += 1u; top %= buf_len; assert (top != bottom); }
        scan_stack[top] = x;
    }
    fn scan_pop() -> uint {
        assert (!scan_stack_empty);
        let x = scan_stack[top];
        if top == bottom {
            scan_stack_empty = true;
        } else { top += buf_len - 1u; top %= buf_len; }
        ret x;
    }
    fn scan_top() -> uint { assert (!scan_stack_empty); ret scan_stack[top]; }
    fn scan_pop_bottom() -> uint {
        assert (!scan_stack_empty);
        let x = scan_stack[bottom];
        if top == bottom {
            scan_stack_empty = true;
        } else { bottom += 1u; bottom %= buf_len; }
        ret x;
    }
    fn advance_right() {
        right += 1u;
        right %= buf_len;
        assert (right != left);
    }
    fn advance_left(x: token, L: int) {
        log #fmt["advnce_left [%u,%u], sizeof(%u)=%d", left, right, left, L];
        if L >= 0 {
            self.print(x, L);
            alt x {
              BREAK(b) { left_total += b.blank_space; }
              STRING(_, len) { assert (len == L); left_total += len; }
              _ { }
            }
            if left != right {
                left += 1u;
                left %= buf_len;
                self.advance_left(token[left], size[left]);
            }
        }
    }
    fn check_stack(k: int) {
        if !scan_stack_empty {
            let x = self.scan_top();
            alt token[x] {
              BEGIN(b) {
                if k > 0 {
                    size[self.scan_pop()] = size[x] + right_total;
                    self.check_stack(k - 1);
                }
              }
              END. {
                // paper says + not =, but that makes no sense.

                size[self.scan_pop()] = 1;
                self.check_stack(k + 1);
              }
              _ {
                size[self.scan_pop()] = size[x] + right_total;
                if k > 0 { self.check_stack(k); }
              }
            }
        }
    }
    fn print_newline(amount: int) {
        log #fmt["NEWLINE %d", amount];
        out.write_str("\n");
        pending_indentation = 0;
        self.indent(amount);
    }
    fn indent(amount: int) {
        log #fmt["INDENT %d", amount];
        pending_indentation += amount;
    }
    fn top() -> print_stack_elt {
        let n = vec::len(print_stack);
        let top: print_stack_elt = {offset: 0, pbreak: broken(inconsistent)};
        if n != 0u { top = print_stack[n - 1u]; }
        ret top;
    }
    fn write_str(s: str) {
        while pending_indentation > 0 {
            out.write_str(" ");
            pending_indentation -= 1;
        }
        out.write_str(s);
    }
    fn print(x: token, L: int) {
        log #fmt["print %s %d (remaining line space=%d)", tok_str(x), L,
                 space];
        log buf_str(token, size, left, right, 6u);
        alt x {
          BEGIN(b) {
            if L > space {
                let col = margin - space + b.offset;
                log #fmt["print BEGIN -> push broken block at col %d", col];
                print_stack += [{offset: col, pbreak: broken(b.breaks)}];
            } else {
                log "print BEGIN -> push fitting block";
                print_stack += [{offset: 0, pbreak: fits}];
            }
          }
          END. {
            log "print END -> pop END";
            assert (vec::len(print_stack) != 0u);
            vec::pop(print_stack);
          }
          BREAK(b) {
            let top = self.top();
            alt top.pbreak {
              fits. {
                log "print BREAK in fitting block";
                space -= b.blank_space;
                self.indent(b.blank_space);
              }
              broken(consistent.) {
                log "print BREAK in consistent block";
                self.print_newline(top.offset + b.offset);
                space = margin - (top.offset + b.offset);
              }
              broken(inconsistent.) {
                if L > space {
                    log "print BREAK w/ newline in inconsistent";
                    self.print_newline(top.offset + b.offset);
                    space = margin - (top.offset + b.offset);
                } else {
                    log "print BREAK w/o newline in inconsistent";
                    self.indent(b.blank_space);
                    space -= b.blank_space;
                }
              }
            }
          }
          STRING(s, len) {
            log "print STRING";
            assert (L == len);
            // assert L <= space;

            space -= len;
            self.write_str(s);
          }
          EOF. {
            // EOF should never get here.

            fail;
          }
        }
    }
}


// Convenience functions to talk to the printer.
fn box(p: printer, indent: uint, b: breaks) {
    p.pretty_print(BEGIN({offset: indent as int, breaks: b}));
}

fn ibox(p: printer, indent: uint) { box(p, indent, inconsistent); }

fn cbox(p: printer, indent: uint) { box(p, indent, consistent); }

fn break_offset(p: printer, n: uint, off: int) {
    p.pretty_print(BREAK({offset: off, blank_space: n as int}));
}

fn end(p: printer) { p.pretty_print(END); }

fn eof(p: printer) { p.pretty_print(EOF); }

fn word(p: printer, wrd: str) {
    p.pretty_print(STRING(wrd, str::char_len(wrd) as int));
}

fn huge_word(p: printer, wrd: str) {
    p.pretty_print(STRING(wrd, size_infinity));
}

fn zero_word(p: printer, wrd: str) { p.pretty_print(STRING(wrd, 0)); }

fn spaces(p: printer, n: uint) { break_offset(p, n, 0); }

fn zerobreak(p: printer) { spaces(p, 0u); }

fn space(p: printer) { spaces(p, 1u); }

fn hardbreak(p: printer) { spaces(p, size_infinity as uint); }

fn hardbreak_tok_offset(off: int) -> token {
    ret BREAK({offset: off, blank_space: size_infinity});
}

fn hardbreak_tok() -> token { ret hardbreak_tok_offset(0); }


//
// Local Variables:
// mode: rust
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:
//

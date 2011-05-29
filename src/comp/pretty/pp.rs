import std::io;
import std::vec;
import std::str;

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
 */

tag breaks { consistent; inconsistent; }
type break_t = rec(int offset, int blank_space);
type begin_t = rec(int offset, breaks breaks);

tag token {
    STRING(str);
    BREAK(break_t);
    BEGIN(begin_t);
    END;
    EOF;
}


fn tok_str(token t) -> str {
    alt (t) {
        case (STRING(?s)) { ret "STR(" + s + ")"; }
        case (BREAK(_)) { ret "BREAK"; }
        case (BEGIN(_)) { ret "BEGIN"; }
        case (END) { ret "END"; }
        case (EOF) { ret "EOF"; }
    }
}

fn buf_str(vec[token] toks, vec[int] szs,
           uint left, uint right, uint lim) -> str {
    auto n = vec::len(toks);
    assert n == vec::len(szs);
    auto i = left;
    auto L = lim;
    auto s = "[";
    while (i != right && L != 0u) {
        L -= 1u;
        if (i != left) {
            s += ", ";
        }
        s += #fmt("%d=%s", szs.(i), tok_str(toks.(i)));
        i += 1u;
        i %= n;
    }
    s += "]";
    ret s;
}


tag print_stack_break { fits; broken(breaks); }
type print_stack_elt = rec(int offset, print_stack_break pbreak);

const int size_infinity = 0xffff;

fn mk_printer(io::writer out, uint linewidth) -> printer {

    // Yes 3, it makes the ring buffers big enough to never
    // fall behind.
    let uint n = 3u * linewidth;

    log #fmt("mk_printer %u", linewidth);

    let vec[token] token = vec::init_elt[token](EOF, n);
    let vec[int] size = vec::init_elt[int](0, n);
    let vec[uint] scan_stack = vec::init_elt[uint](0u, n);
    let vec[print_stack_elt] print_stack = [];

    ret printer(out,
                n,
                linewidth as int, // margin
                linewidth as int, // space
                0u,               // left
                0u,               // right
                token,
                size,
                0,                // left_total
                0,                // right_total
                scan_stack,
                true,             // scan_stack_empty
                0u,               // top
                0u,               // bottom
                print_stack);
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

obj printer(io::writer out,
            uint buf_len,
            mutable int margin,        // width of lines we're constrained to
            mutable int space,         // number of spaces left on line

            mutable uint left,         // index of left side of input stream
            mutable uint right,        // index of right side of input stream
            mutable vec[token] token,  // ring-buffer stream goes through
            mutable vec[int] size,     // ring-buffer of calculated sizes
            mutable int left_total,    // running size of stream "...left"
            mutable int right_total,   // running size of stream "...right"

            // pseudo-stack, really a ring too. Holds the primary-ring-buffers
            // index of the BEGIN that started the current block, possibly
            // with the most recent BREAK after that BEGIN (if there is any)
            // on top of it. Stuff is flushed off the bottom as it becomes
            // irrelevant due to the primary ring-buffer advancing.

            mutable vec[uint] scan_stack,
            mutable bool scan_stack_empty, // top==bottom disambiguator
            mutable uint top,              // index of top of scan_stack
            mutable uint bottom,           // index of bottom of scan_stack

            // stack of blocks-in-progress being flushed by print
            mutable vec[print_stack_elt] print_stack
            ) {


    fn pretty_print(token t) {

        log #fmt("pp [%u,%u]", left, right);
        alt (t) {

            case (EOF) {
                if (!scan_stack_empty) {
                    self.check_stack(0);
                    self.advance_left(token.(left), size.(left));
                }
                self.indent(0);
            }

            case (BEGIN(?b)) {
                if (scan_stack_empty) {
                    left_total = 1;
                    right_total = 1;
                    left = 0u;
                    right = 0u;
                } else {
                    self.advance_right();
                }
                log #fmt("pp BEGIN/buffer [%u,%u]", left, right);
                token.(right) = t;
                size.(right) = -right_total;
                self.scan_push(right);
            }

            case (END) {
                if (scan_stack_empty) {
                    log #fmt("pp END/print [%u,%u]", left, right);
                    self.print(t, 0);
                } else {
                    log #fmt("pp END/buffer [%u,%u]", left, right);
                    self.advance_right();
                    token.(right) = t;
                    size.(right) = -1;
                    self.scan_push(right);
                }
            }

            case (BREAK(?b)) {
                if (scan_stack_empty) {
                    left_total = 1;
                    right_total = 1;
                    left = 0u;
                    right = 0u;
                } else {
                    self.advance_right();
                }
                log #fmt("pp BREAK/buffer [%u,%u]", left, right);
                self.check_stack(0);
                self.scan_push(right);
                token.(right) = t;
                size.(right) = -right_total;
                right_total += b.blank_space;
            }

            case (STRING(?s)) {
                auto len = str::char_len(s) as int;
                if (scan_stack_empty) {
                    log #fmt("pp STRING/print [%u,%u]", left, right);
                    self.print(t, len);
                } else {
                    log #fmt("pp STRING/buffer [%u,%u]", left, right);
                    self.advance_right();
                    token.(right) = t;
                    size.(right) = len;
                    right_total += len;
                    self.check_stream();
                }
            }
        }
    }

    fn check_stream() {
        log #fmt("check_stream [%u, %u] with left_total=%d, right_total=%d",
                     left, right, left_total, right_total);;
        if (right_total - left_total > space) {
            log #fmt("scan window is %d, longer than space on line (%d)",
                         right_total - left_total, space);
            if (!scan_stack_empty) {
                if (left == scan_stack.(bottom)) {
                    log #fmt("setting %u to infinity and popping", left);
                    size.(self.scan_pop_bottom()) = size_infinity;
                }
            }
            self.advance_left(token.(left), size.(left));
            if (left != right) {
                self.check_stream();
            }
        }
    }

    fn scan_push(uint x) {
        log #fmt("scan_push %u", x);
        if (scan_stack_empty) {
            scan_stack_empty = false;
        } else {
            top += 1u;
            top %= buf_len;
            assert top != bottom;
        }
        scan_stack.(top) = x;
    }

    fn scan_pop() -> uint {
        assert !scan_stack_empty;
        auto x = scan_stack.(top);
        if (top == bottom) {
            scan_stack_empty = true;
        } else {
            top += (buf_len - 1u);
            top %= buf_len;
        }
        ret x;
    }

    fn scan_top() -> uint {
        assert !scan_stack_empty;
        ret scan_stack.(top);
    }

    fn scan_pop_bottom() -> uint {
        assert !scan_stack_empty;
        auto x = scan_stack.(bottom);
        if (top == bottom) {
            scan_stack_empty = true;
        } else {
            bottom += 1u;
            bottom %= buf_len;
        }
        ret x;
    }

    fn advance_right() {
        right += 1u;
        right %= buf_len;
        assert right != left;
    }

    fn advance_left(token x, int L) {
        log #fmt("advnce_left [%u,%u], sizeof(%u)=%d", left, right, left, L);
        if (L >= 0) {
            self.print(x, L);
            alt (x) {
                case (BREAK(?b)) {
                    left_total += b.blank_space;
                }
                case (STRING(?s)) {
                    // I think? paper says '1' here but 1 and L look same in
                    // it.
                    left_total += L;
                }
                case (_) {}
            }
            if (left != right) {
                left += 1u;
                left %= buf_len;
                self.advance_left(token.(left), size.(left));
            }
        }
    }

    fn check_stack(int k) {
        if (!scan_stack_empty) {
            auto x = self.scan_top();
            alt (token.(x)) {
                case (BEGIN(?b)) {
                    if (k > 0) {
                        size.(self.scan_pop()) = size.(x) + right_total;
                        self.check_stack(k - 1);
                    }
                }
                case (END) {
                    // paper says + not =, but that makes no sense.
                    size.(self.scan_pop()) = 1;
                    self.check_stack(k + 1);
                }
                case (_) {
                    size.(self.scan_pop()) = size.(x) + right_total;
                    if (k > 0) {
                        self.check_stack(k);
                    }
                }
            }
        }
    }

    fn print_newline(int amount) {
        log #fmt("NEWLINE %d", amount);
        out.write_str("\n");
        self.indent(amount);
    }

    fn indent(int amount) {
        log #fmt("INDENT %d", amount);
        auto u = 0;
        while (u < amount) {
            out.write_str(" ");
            u += 1;
        }
    }

    fn print(token x, int L) {
        log #fmt("print %s %d (remaining line space=%d)", tok_str(x), L, space);
        log buf_str(token, size, left, right, 6u);
        alt (x) {
            case (BEGIN(?b)) {
                if (L > space) {
                    auto col = (margin - space) + b.offset;
                    log #fmt("print BEGIN -> push broken block at col %d", col);
                    vec::push(print_stack,
                              rec(offset = col,
                                  pbreak = broken(b.breaks)));
                } else {
                    log "print BEGIN -> push fitting block";
                    vec::push(print_stack,
                              rec(offset = 0,
                                  pbreak = fits));
                }
            }

            case (END) {
                log "print END -> pop END";
                assert vec::len(print_stack) != 0u;
                vec::pop(print_stack);
            }

            case (BREAK(?b)) {

                auto n = vec::len(print_stack);
                let print_stack_elt top =
                    rec(offset=0, pbreak=broken(inconsistent));;
                if (n != 0u) {
                    top = print_stack.(n - 1u);
                }

                alt (top.pbreak) {
                    case (fits) {
                        log "print BREAK in fitting block";
                        space -= b.blank_space;
                        self.indent(b.blank_space);
                    }

                    case (broken(consistent)) {
                        log "print BREAK in consistent block";
                        self.print_newline(top.offset + b.offset);
                        space = margin - (top.offset + b.offset);
                    }

                    case (broken(inconsistent)) {
                        if (L > space) {
                            log "print BREAK w/ newline in inconsistent block";
                            self.print_newline(top.offset + b.offset);
                            space = margin - (top.offset + b.offset);
                        } else {
                            log "print BREAK w/o newline in inconsistent block";
                            self.indent(b.blank_space);
                            space -= b.blank_space;
                        }
                    }
                }
            }

            case (STRING(?s)) {
                log "print STRING";
                assert L as uint == str::char_len(s);
                // assert L <= space;
                space -= L;
                out.write_str(s);
            }

            case (EOF) {
                // EOF should never get here.
                fail;
            }
        }
    }
}


// Convenience functions to talk to the printer.

fn ibox(printer p, uint indent) {
    p.pretty_print(BEGIN(rec(offset = indent as int,
                             breaks = inconsistent)));
}

fn cbox(printer p, uint indent) {
    p.pretty_print(BEGIN(rec(offset = indent as int,
                             breaks = consistent)));
}


fn break_offset(printer p, uint n, int off) {
    p.pretty_print(BREAK(rec(offset = off,
                             blank_space = n as int)));
}

fn end(printer p) { p.pretty_print(END); }
fn eof(printer p) { p.pretty_print(EOF); }
fn wrd(printer p, str wrd) { p.pretty_print(STRING(wrd)); }
fn spaces(printer p, uint n) { break_offset(p, n, 0); }
fn space(printer p) { spaces(p, 1u); }
fn hardbreak(printer p) { spaces(p, 0xffffu); }



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

// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
 * it's 32 years old. What can I say?
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
 * I also implemented a small change in the String token, in that I store an
 * explicit length for the string. For most tokens this is just the length of
 * the accompanying string. But it's necessary to permit it to differ, for
 * encoding things that are supposed to "go on their own line" -- certain
 * classes of comment and blank-line -- where relying on adjacent
 * hardbreak-like Break tokens with long blankness indication doesn't actually
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

use std::io;
use std::vec;

#[deriving(Clone, Eq)]
pub enum Breaks {
    Consistent,
    Inconsistent,
}

#[deriving(Clone)]
pub struct BreakToken {
    offset: int,
    blank_space: int
}

#[deriving(Clone)]
pub struct BeginToken {
    offset: int,
    breaks: Breaks
}

#[deriving(Clone)]
pub enum Token {
    String(~str, int),
    Break(BreakToken),
    Begin(BeginToken),
    End,
    Eof,
}

impl Token {
    pub fn is_eof(&self) -> bool {
        match *self { Eof => true, _ => false }
    }

    pub fn is_hardbreak_tok(&self) -> bool {
        match *self {
            Break(BreakToken {
                offset: 0,
                blank_space: bs
            }) if bs == SIZE_INFINITY =>
                true,
            _ =>
                false
        }
    }
}

pub fn tok_str(t: Token) -> ~str {
    match t {
        String(s, len) => return format!("STR({},{})", s, len),
        Break(_) => return ~"BREAK",
        Begin(_) => return ~"BEGIN",
        End => return ~"END",
        Eof => return ~"EOF"
    }
}

pub fn buf_str(toks: ~[Token], szs: ~[int], left: uint, right: uint,
               lim: uint) -> ~str {
    let n = toks.len();
    assert_eq!(n, szs.len());
    let mut i = left;
    let mut L = lim;
    let mut s = ~"[";
    while i != right && L != 0u {
        L -= 1u;
        if i != left {
            s.push_str(", ");
        }
        s.push_str(format!("{}={}", szs[i], tok_str(toks[i].clone())));
        i += 1u;
        i %= n;
    }
    s.push_char(']');
    return s;
}

enum PrintStackBreak {
    Fits,
    Broken(Breaks),
}

struct PrintStackElem {
    offset: int,
    pbreak: PrintStackBreak
}

static SIZE_INFINITY: int = 0xffff;

pub fn mk_printer(out: ~io::Writer, linewidth: uint) -> Printer {
    // Yes 3, it makes the ring buffers big enough to never
    // fall behind.
    let n: uint = 3 * linewidth;
    debug!("mk_printer {}", linewidth);
    let token: ~[Token] = vec::from_elem(n, Eof);
    let size: ~[int] = vec::from_elem(n, 0);
    let scan_stack: ~[uint] = vec::from_elem(n, 0u);
    Printer {
        out: out,
        buf_len: n,
        margin: linewidth as int,
        space: linewidth as int,
        left: 0,
        right: 0,
        token: token,
        size: size,
        left_total: 0,
        right_total: 0,
        scan_stack: scan_stack,
        scan_stack_empty: true,
        top: 0,
        bottom: 0,
        print_stack: ~[],
        pending_indentation: 0
    }
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
 * Tokens are String, Break, and Begin/End to delimit blocks.
 *
 * Begin tokens can carry an offset, saying "how far to indent when you break
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
 * each token. Why calculated? Because for Begin/End pairs, the "size"
 * includes everything betwen the pair. That is, the "size" of Begin is
 * actually the sum of the sizes of everything between Begin and the paired
 * End that follows. Since that is arbitrarily far in the future, 'size' is
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
 * So SCAN takes input and buffers tokens and pending calculations, while
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
pub struct Printer {
    out: ~io::Writer,
    buf_len: uint,
    margin: int, // width of lines we're constrained to
    space: int, // number of spaces left on line
    left: uint, // index of left side of input stream
    right: uint, // index of right side of input stream
    token: ~[Token], // ring-buffr stream goes through
    size: ~[int], // ring-buffer of calculated sizes
    left_total: int, // running size of stream "...left"
    right_total: int, // running size of stream "...right"
    // pseudo-stack, really a ring too. Holds the
    // primary-ring-buffers index of the Begin that started the
    // current block, possibly with the most recent Break after that
    // Begin (if there is any) on top of it. Stuff is flushed off the
    // bottom as it becomes irrelevant due to the primary ring-buffer
    // advancing.
    scan_stack: ~[uint],
    scan_stack_empty: bool, // top==bottom disambiguator
    top: uint, // index of top of scan_stack
    bottom: uint, // index of bottom of scan_stack
    // stack of blocks-in-progress being flushed by print
    print_stack: ~[PrintStackElem],
    // buffered indentation to avoid writing trailing whitespace
    pending_indentation: int,
}

impl Printer {
    pub fn last_token(&mut self) -> Token {
        self.token[self.right].clone()
    }
    // be very careful with this!
    pub fn replace_last_token(&mut self, t: Token) {
        self.token[self.right] = t;
    }
    pub fn pretty_print(&mut self, t: Token) -> io::IoResult<()> {
        debug!("pp ~[{},{}]", self.left, self.right);
        match t {
          Eof => {
            if !self.scan_stack_empty {
                self.check_stack(0);
                let left = self.token[self.left].clone();
                try!(self.advance_left(left, self.size[self.left]));
            }
            self.indent(0);
            Ok(())
          }
          Begin(b) => {
            if self.scan_stack_empty {
                self.left_total = 1;
                self.right_total = 1;
                self.left = 0u;
                self.right = 0u;
            } else { self.advance_right(); }
            debug!("pp Begin({})/buffer ~[{},{}]",
                   b.offset, self.left, self.right);
            self.token[self.right] = t;
            self.size[self.right] = -self.right_total;
            self.scan_push(self.right);
            Ok(())
          }
          End => {
            if self.scan_stack_empty {
                debug!("pp End/print ~[{},{}]", self.left, self.right);
                self.print(t, 0)
            } else {
                debug!("pp End/buffer ~[{},{}]", self.left, self.right);
                self.advance_right();
                self.token[self.right] = t;
                self.size[self.right] = -1;
                self.scan_push(self.right);
                Ok(())
            }
          }
          Break(b) => {
            if self.scan_stack_empty {
                self.left_total = 1;
                self.right_total = 1;
                self.left = 0u;
                self.right = 0u;
            } else { self.advance_right(); }
            debug!("pp Break({})/buffer ~[{},{}]",
                   b.offset, self.left, self.right);
            self.check_stack(0);
            self.scan_push(self.right);
            self.token[self.right] = t;
            self.size[self.right] = -self.right_total;
            self.right_total += b.blank_space;
            Ok(())
          }
          String(ref s, len) => {
            if self.scan_stack_empty {
                debug!("pp String('{}')/print ~[{},{}]",
                       *s, self.left, self.right);
                self.print(t.clone(), len)
            } else {
                debug!("pp String('{}')/buffer ~[{},{}]",
                       *s, self.left, self.right);
                self.advance_right();
                self.token[self.right] = t.clone();
                self.size[self.right] = len;
                self.right_total += len;
                self.check_stream()
            }
          }
        }
    }
    pub fn check_stream(&mut self) -> io::IoResult<()> {
        debug!("check_stream ~[{}, {}] with left_total={}, right_total={}",
               self.left, self.right, self.left_total, self.right_total);
        if self.right_total - self.left_total > self.space {
            debug!("scan window is {}, longer than space on line ({})",
                   self.right_total - self.left_total, self.space);
            if !self.scan_stack_empty {
                if self.left == self.scan_stack[self.bottom] {
                    debug!("setting {} to infinity and popping", self.left);
                    self.size[self.scan_pop_bottom()] = SIZE_INFINITY;
                }
            }
            let left = self.token[self.left].clone();
            try!(self.advance_left(left, self.size[self.left]));
            if self.left != self.right {
                try!(self.check_stream());
            }
        }
        Ok(())
    }
    pub fn scan_push(&mut self, x: uint) {
        debug!("scan_push {}", x);
        if self.scan_stack_empty {
            self.scan_stack_empty = false;
        } else {
            self.top += 1u;
            self.top %= self.buf_len;
            assert!((self.top != self.bottom));
        }
        self.scan_stack[self.top] = x;
    }
    pub fn scan_pop(&mut self) -> uint {
        assert!((!self.scan_stack_empty));
        let x = self.scan_stack[self.top];
        if self.top == self.bottom {
            self.scan_stack_empty = true;
        } else { self.top += self.buf_len - 1u; self.top %= self.buf_len; }
        return x;
    }
    pub fn scan_top(&mut self) -> uint {
        assert!((!self.scan_stack_empty));
        return self.scan_stack[self.top];
    }
    pub fn scan_pop_bottom(&mut self) -> uint {
        assert!((!self.scan_stack_empty));
        let x = self.scan_stack[self.bottom];
        if self.top == self.bottom {
            self.scan_stack_empty = true;
        } else { self.bottom += 1u; self.bottom %= self.buf_len; }
        return x;
    }
    pub fn advance_right(&mut self) {
        self.right += 1u;
        self.right %= self.buf_len;
        assert!((self.right != self.left));
    }
    pub fn advance_left(&mut self, x: Token, L: int) -> io::IoResult<()> {
        debug!("advnce_left ~[{},{}], sizeof({})={}", self.left, self.right,
               self.left, L);
        if L >= 0 {
            let ret = self.print(x.clone(), L);
            match x {
              Break(b) => self.left_total += b.blank_space,
              String(_, len) => {
                assert_eq!(len, L); self.left_total += len;
              }
              _ => ()
            }
            if self.left != self.right {
                self.left += 1u;
                self.left %= self.buf_len;
                let left = self.token[self.left].clone();
                try!(self.advance_left(left, self.size[self.left]));
            }
            ret
        } else {
            Ok(())
        }
    }
    pub fn check_stack(&mut self, k: int) {
        if !self.scan_stack_empty {
            let x = self.scan_top();
            match self.token[x] {
              Begin(_) => {
                if k > 0 {
                    self.size[self.scan_pop()] = self.size[x] +
                        self.right_total;
                    self.check_stack(k - 1);
                }
              }
              End => {
                // paper says + not =, but that makes no sense.
                self.size[self.scan_pop()] = 1;
                self.check_stack(k + 1);
              }
              _ => {
                self.size[self.scan_pop()] = self.size[x] + self.right_total;
                if k > 0 { self.check_stack(k); }
              }
            }
        }
    }
    pub fn print_newline(&mut self, amount: int) -> io::IoResult<()> {
        debug!("NEWLINE {}", amount);
        let ret = write!(self.out, "\n");
        self.pending_indentation = 0;
        self.indent(amount);
        return ret;
    }
    pub fn indent(&mut self, amount: int) {
        debug!("INDENT {}", amount);
        self.pending_indentation += amount;
    }
    pub fn get_top(&mut self) -> PrintStackElem {
        let print_stack = &mut self.print_stack;
        let n = print_stack.len();
        if n != 0u {
            print_stack[n - 1u]
        } else {
            PrintStackElem {
                offset: 0,
                pbreak: Broken(Inconsistent)
            }
        }
    }
    pub fn print_str(&mut self, s: &str) -> io::IoResult<()> {
        while self.pending_indentation > 0 {
            try!(write!(self.out, " "));
            self.pending_indentation -= 1;
        }
        write!(self.out, "{}", s)
    }
    pub fn print(&mut self, x: Token, L: int) -> io::IoResult<()> {
        debug!("print {} {} (remaining line space={})", tok_str(x.clone()), L,
               self.space);
        debug!("{}", buf_str(self.token.clone(),
                             self.size.clone(),
                             self.left,
                             self.right,
                             6));
        match x {
          Begin(b) => {
            if L > self.space {
                let col = self.margin - self.space + b.offset;
                debug!("print Begin -> push broken block at col {}", col);
                self.print_stack.push(PrintStackElem {
                    offset: col,
                    pbreak: Broken(b.breaks)
                });
            } else {
                debug!("print Begin -> push fitting block");
                self.print_stack.push(PrintStackElem {
                    offset: 0,
                    pbreak: Fits
                });
            }
            Ok(())
          }
          End => {
            debug!("print End -> pop End");
            let print_stack = &mut self.print_stack;
            assert!((print_stack.len() != 0u));
            print_stack.pop().unwrap();
            Ok(())
          }
          Break(b) => {
            let top = self.get_top();
            match top.pbreak {
              Fits => {
                debug!("print Break({}) in fitting block", b.blank_space);
                self.space -= b.blank_space;
                self.indent(b.blank_space);
                Ok(())
              }
              Broken(Consistent) => {
                debug!("print Break({}+{}) in consistent block",
                       top.offset, b.offset);
                let ret = self.print_newline(top.offset + b.offset);
                self.space = self.margin - (top.offset + b.offset);
                ret
              }
              Broken(Inconsistent) => {
                if L > self.space {
                    debug!("print Break({}+{}) w/ newline in inconsistent",
                           top.offset, b.offset);
                    let ret = self.print_newline(top.offset + b.offset);
                    self.space = self.margin - (top.offset + b.offset);
                    ret
                } else {
                    debug!("print Break({}) w/o newline in inconsistent",
                           b.blank_space);
                    self.indent(b.blank_space);
                    self.space -= b.blank_space;
                    Ok(())
                }
              }
            }
          }
          String(s, len) => {
            debug!("print String({})", s);
            assert_eq!(L, len);
            // assert!(L <= space);
            self.space -= len;
            self.print_str(s)
          }
          Eof => {
            // Eof should never get here.
            fail!();
          }
        }
    }
}

// Convenience functions to talk to the printer.
//
// "raw box"
pub fn rbox(p: &mut Printer, indent: uint, b: Breaks) -> io::IoResult<()> {
    p.pretty_print(Begin(BeginToken {
        offset: indent as int,
        breaks: b
    }))
}

pub fn ibox(p: &mut Printer, indent: uint) -> io::IoResult<()> {
    rbox(p, indent, Inconsistent)
}

pub fn cbox(p: &mut Printer, indent: uint) -> io::IoResult<()> {
    rbox(p, indent, Consistent)
}

pub fn break_offset(p: &mut Printer, n: uint, off: int) -> io::IoResult<()> {
    p.pretty_print(Break(BreakToken {
        offset: off,
        blank_space: n as int
    }))
}

pub fn end(p: &mut Printer) -> io::IoResult<()> { p.pretty_print(End) }

pub fn eof(p: &mut Printer) -> io::IoResult<()> { p.pretty_print(Eof) }

pub fn word(p: &mut Printer, wrd: &str) -> io::IoResult<()> {
    p.pretty_print(String(/* bad */ wrd.to_str(), wrd.len() as int))
}

pub fn huge_word(p: &mut Printer, wrd: &str) -> io::IoResult<()> {
    p.pretty_print(String(/* bad */ wrd.to_str(), SIZE_INFINITY))
}

pub fn zero_word(p: &mut Printer, wrd: &str) -> io::IoResult<()> {
    p.pretty_print(String(/* bad */ wrd.to_str(), 0))
}

pub fn spaces(p: &mut Printer, n: uint) -> io::IoResult<()> {
    break_offset(p, n, 0)
}

pub fn zerobreak(p: &mut Printer) -> io::IoResult<()> {
    spaces(p, 0u)
}

pub fn space(p: &mut Printer) -> io::IoResult<()> {
    spaces(p, 1u)
}

pub fn hardbreak(p: &mut Printer) -> io::IoResult<()> {
    spaces(p, SIZE_INFINITY as uint)
}

pub fn hardbreak_tok_offset(off: int) -> Token {
    Break(BreakToken {offset: off, blank_space: SIZE_INFINITY})
}

pub fn hardbreak_tok() -> Token { return hardbreak_tok_offset(0); }

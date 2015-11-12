// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! This pretty-printer is a direct reimplementation of Philip Karlton's
//! Mesa pretty-printer, as described in appendix A of
//!
//! ````ignore
//! STAN-CS-79-770: "Pretty Printing", by Derek C. Oppen.
//! Stanford Department of Computer Science, 1979.
//! ````
//!
//! The algorithm's aim is to break a stream into as few lines as possible
//! while respecting the indentation-consistency requirements of the enclosing
//! block, and avoiding breaking at silly places on block boundaries, for
//! example, between "x" and ")" in "x)".
//!
//! I am implementing this algorithm because it comes with 20 pages of
//! documentation explaining its theory, and because it addresses the set of
//! concerns I've seen other pretty-printers fall down on. Weirdly. Even though
//! it's 32 years old. What can I say?
//!
//! Despite some redundancies and quirks in the way it's implemented in that
//! paper, I've opted to keep the implementation here as similar as I can,
//! changing only what was blatantly wrong, a typo, or sufficiently
//! non-idiomatic rust that it really stuck out.
//!
//! In particular you'll see a certain amount of churn related to INTEGER vs.
//! CARDINAL in the Mesa implementation. Mesa apparently interconverts the two
//! somewhat readily? In any case, I've used usize for indices-in-buffers and
//! ints for character-sizes-and-indentation-offsets. This respects the need
//! for ints to "go negative" while carrying a pending-calculation balance, and
//! helps differentiate all the numbers flying around internally (slightly).
//!
//! I also inverted the indentation arithmetic used in the print stack, since
//! the Mesa implementation (somewhat randomly) stores the offset on the print
//! stack in terms of margin-col rather than col itself. I store col.
//!
//! I also implemented a small change in the String token, in that I store an
//! explicit length for the string. For most tokens this is just the length of
//! the accompanying string. But it's necessary to permit it to differ, for
//! encoding things that are supposed to "go on their own line" -- certain
//! classes of comment and blank-line -- where relying on adjacent
//! hardbreak-like Break tokens with long blankness indication doesn't actually
//! work. To see why, consider when there is a "thing that should be on its own
//! line" between two long blocks, say functions. If you put a hardbreak after
//! each function (or before each) and the breaking algorithm decides to break
//! there anyways (because the functions themselves are long) you wind up with
//! extra blank lines. If you don't put hardbreaks you can wind up with the
//! "thing which should be on its own line" not getting its own line in the
//! rare case of "really small functions" or such. This re-occurs with comments
//! and explicit blank lines. So in those cases we use a string with a payload
//! we want isolated to a line and an explicit length that's huge, surrounded
//! by two zero-length breaks. The algorithm will try its best to fit it on a
//! line (which it can't) and so naturally place the content on its own line to
//! avoid combining it with other lines and making matters even worse.

use std::io;
use std::string;

#[derive(Clone, Copy, PartialEq)]
pub enum Breaks {
    Consistent,
    Inconsistent,
}

#[derive(Clone, Copy)]
pub struct BreakToken {
    offset: isize,
    blank_space: isize
}

#[derive(Clone, Copy)]
pub struct BeginToken {
    offset: isize,
    breaks: Breaks
}

#[derive(Clone)]
pub enum Token {
    String(String, isize),
    Break(BreakToken),
    Begin(BeginToken),
    End,
    Eof,
}

impl Token {
    pub fn is_eof(&self) -> bool {
        match *self {
            Token::Eof => true,
            _ => false,
        }
    }

    pub fn is_hardbreak_tok(&self) -> bool {
        match *self {
            Token::Break(BreakToken {
                offset: 0,
                blank_space: bs
            }) if bs == SIZE_INFINITY =>
                true,
            _ =>
                false
        }
    }
}

pub fn tok_str(token: &Token) -> String {
    match *token {
        Token::String(ref s, len) => format!("STR({},{})", s, len),
        Token::Break(_) => "BREAK".to_string(),
        Token::Begin(_) => "BEGIN".to_string(),
        Token::End => "END".to_string(),
        Token::Eof => "EOF".to_string()
    }
}

pub fn buf_str(toks: &[Token],
               szs: &[isize],
               left: usize,
               right: usize,
               lim: usize)
               -> String {
    let n = toks.len();
    assert_eq!(n, szs.len());
    let mut i = left;
    let mut l = lim;
    let mut s = string::String::from("[");
    while i != right && l != 0 {
        l -= 1;
        if i != left {
            s.push_str(", ");
        }
        s.push_str(&format!("{}={}",
                           szs[i],
                           tok_str(&toks[i])));
        i += 1;
        i %= n;
    }
    s.push(']');
    s
}

#[derive(Copy, Clone)]
pub enum PrintStackBreak {
    Fits,
    Broken(Breaks),
}

#[derive(Copy, Clone)]
pub struct PrintStackElem {
    offset: isize,
    pbreak: PrintStackBreak
}

const SIZE_INFINITY: isize = 0xffff;

pub fn mk_printer<'a>(out: Box<io::Write+'a>, linewidth: usize) -> Printer<'a> {
    // Yes 3, it makes the ring buffers big enough to never
    // fall behind.
    let n: usize = 3 * linewidth;
    debug!("mk_printer {}", linewidth);
    let token = vec![Token::Eof; n];
    let size = vec![0_isize; n];
    let scan_stack = vec![0_usize; n];
    Printer {
        out: out,
        buf_len: n,
        margin: linewidth as isize,
        space: linewidth as isize,
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
        print_stack: Vec::new(),
        pending_indentation: 0
    }
}


/// In case you do not have the paper, here is an explanation of what's going
/// on.
///
/// There is a stream of input tokens flowing through this printer.
///
/// The printer buffers up to 3N tokens inside itself, where N is linewidth.
/// Yes, linewidth is chars and tokens are multi-char, but in the worst
/// case every token worth buffering is 1 char long, so it's ok.
///
/// Tokens are String, Break, and Begin/End to delimit blocks.
///
/// Begin tokens can carry an offset, saying "how far to indent when you break
/// inside here", as well as a flag indicating "consistent" or "inconsistent"
/// breaking. Consistent breaking means that after the first break, no attempt
/// will be made to flow subsequent breaks together onto lines. Inconsistent
/// is the opposite. Inconsistent breaking example would be, say:
///
///  foo(hello, there, good, friends)
///
/// breaking inconsistently to become
///
///  foo(hello, there
///      good, friends);
///
/// whereas a consistent breaking would yield:
///
///  foo(hello,
///      there
///      good,
///      friends);
///
/// That is, in the consistent-break blocks we value vertical alignment
/// more than the ability to cram stuff onto a line. But in all cases if it
/// can make a block a one-liner, it'll do so.
///
/// Carrying on with high-level logic:
///
/// The buffered tokens go through a ring-buffer, 'tokens'. The 'left' and
/// 'right' indices denote the active portion of the ring buffer as well as
/// describing hypothetical points-in-the-infinite-stream at most 3N tokens
/// apart (i.e. "not wrapped to ring-buffer boundaries"). The paper will switch
/// between using 'left' and 'right' terms to denote the wrapped-to-ring-buffer
/// and point-in-infinite-stream senses freely.
///
/// There is a parallel ring buffer, 'size', that holds the calculated size of
/// each token. Why calculated? Because for Begin/End pairs, the "size"
/// includes everything between the pair. That is, the "size" of Begin is
/// actually the sum of the sizes of everything between Begin and the paired
/// End that follows. Since that is arbitrarily far in the future, 'size' is
/// being rewritten regularly while the printer runs; in fact most of the
/// machinery is here to work out 'size' entries on the fly (and give up when
/// they're so obviously over-long that "infinity" is a good enough
/// approximation for purposes of line breaking).
///
/// The "input side" of the printer is managed as an abstract process called
/// SCAN, which uses 'scan_stack', 'scan_stack_empty', 'top' and 'bottom', to
/// manage calculating 'size'. SCAN is, in other words, the process of
/// calculating 'size' entries.
///
/// The "output side" of the printer is managed by an abstract process called
/// PRINT, which uses 'print_stack', 'margin' and 'space' to figure out what to
/// do with each token/size pair it consumes as it goes. It's trying to consume
/// the entire buffered window, but can't output anything until the size is >=
/// 0 (sizes are set to negative while they're pending calculation).
///
/// So SCAN takes input and buffers tokens and pending calculations, while
/// PRINT gobbles up completed calculations and tokens from the buffer. The
/// theory is that the two can never get more than 3N tokens apart, because
/// once there's "obviously" too much data to fit on a line, in a size
/// calculation, SCAN will write "infinity" to the size and let PRINT consume
/// it.
///
/// In this implementation (following the paper, again) the SCAN process is
/// the method called 'pretty_print', and the 'PRINT' process is the method
/// called 'print'.
pub struct Printer<'a> {
    pub out: Box<io::Write+'a>,
    buf_len: usize,
    /// Width of lines we're constrained to
    margin: isize,
    /// Number of spaces left on line
    space: isize,
    /// Index of left side of input stream
    left: usize,
    /// Index of right side of input stream
    right: usize,
    /// Ring-buffer stream goes through
    token: Vec<Token> ,
    /// Ring-buffer of calculated sizes
    size: Vec<isize> ,
    /// Running size of stream "...left"
    left_total: isize,
    /// Running size of stream "...right"
    right_total: isize,
    /// Pseudo-stack, really a ring too. Holds the
    /// primary-ring-buffers index of the Begin that started the
    /// current block, possibly with the most recent Break after that
    /// Begin (if there is any) on top of it. Stuff is flushed off the
    /// bottom as it becomes irrelevant due to the primary ring-buffer
    /// advancing.
    scan_stack: Vec<usize> ,
    /// Top==bottom disambiguator
    scan_stack_empty: bool,
    /// Index of top of scan_stack
    top: usize,
    /// Index of bottom of scan_stack
    bottom: usize,
    /// Stack of blocks-in-progress being flushed by print
    print_stack: Vec<PrintStackElem> ,
    /// Buffered indentation to avoid writing trailing whitespace
    pending_indentation: isize,
}

impl<'a> Printer<'a> {
    pub fn last_token(&mut self) -> Token {
        self.token[self.right].clone()
    }
    // be very careful with this!
    pub fn replace_last_token(&mut self, t: Token) {
        self.token[self.right] = t;
    }
    pub fn pretty_print(&mut self, token: Token) -> io::Result<()> {
        debug!("pp Vec<{},{}>", self.left, self.right);
        match token {
          Token::Eof => {
            if !self.scan_stack_empty {
                self.check_stack(0);
                try!(self.advance_left());
            }
            self.indent(0);
            Ok(())
          }
          Token::Begin(b) => {
            if self.scan_stack_empty {
                self.left_total = 1;
                self.right_total = 1;
                self.left = 0;
                self.right = 0;
            } else { self.advance_right(); }
            debug!("pp Begin({})/buffer Vec<{},{}>",
                   b.offset, self.left, self.right);
            self.token[self.right] = token;
            self.size[self.right] = -self.right_total;
            let right = self.right;
            self.scan_push(right);
            Ok(())
          }
          Token::End => {
            if self.scan_stack_empty {
                debug!("pp End/print Vec<{},{}>", self.left, self.right);
                self.print(token, 0)
            } else {
                debug!("pp End/buffer Vec<{},{}>", self.left, self.right);
                self.advance_right();
                self.token[self.right] = token;
                self.size[self.right] = -1;
                let right = self.right;
                self.scan_push(right);
                Ok(())
            }
          }
          Token::Break(b) => {
            if self.scan_stack_empty {
                self.left_total = 1;
                self.right_total = 1;
                self.left = 0;
                self.right = 0;
            } else { self.advance_right(); }
            debug!("pp Break({})/buffer Vec<{},{}>",
                   b.offset, self.left, self.right);
            self.check_stack(0);
            let right = self.right;
            self.scan_push(right);
            self.token[self.right] = token;
            self.size[self.right] = -self.right_total;
            self.right_total += b.blank_space;
            Ok(())
          }
          Token::String(s, len) => {
            if self.scan_stack_empty {
                debug!("pp String('{}')/print Vec<{},{}>",
                       s, self.left, self.right);
                self.print(Token::String(s, len), len)
            } else {
                debug!("pp String('{}')/buffer Vec<{},{}>",
                       s, self.left, self.right);
                self.advance_right();
                self.token[self.right] = Token::String(s, len);
                self.size[self.right] = len;
                self.right_total += len;
                self.check_stream()
            }
          }
        }
    }
    pub fn check_stream(&mut self) -> io::Result<()> {
        debug!("check_stream Vec<{}, {}> with left_total={}, right_total={}",
               self.left, self.right, self.left_total, self.right_total);
        if self.right_total - self.left_total > self.space {
            debug!("scan window is {}, longer than space on line ({})",
                   self.right_total - self.left_total, self.space);
            if !self.scan_stack_empty {
                if self.left == self.scan_stack[self.bottom] {
                    debug!("setting {} to infinity and popping", self.left);
                    let scanned = self.scan_pop_bottom();
                    self.size[scanned] = SIZE_INFINITY;
                }
            }
            try!(self.advance_left());
            if self.left != self.right {
                try!(self.check_stream());
            }
        }
        Ok(())
    }
    pub fn scan_push(&mut self, x: usize) {
        debug!("scan_push {}", x);
        if self.scan_stack_empty {
            self.scan_stack_empty = false;
        } else {
            self.top += 1;
            self.top %= self.buf_len;
            assert!((self.top != self.bottom));
        }
        self.scan_stack[self.top] = x;
    }
    pub fn scan_pop(&mut self) -> usize {
        assert!((!self.scan_stack_empty));
        let x = self.scan_stack[self.top];
        if self.top == self.bottom {
            self.scan_stack_empty = true;
        } else {
            self.top += self.buf_len - 1; self.top %= self.buf_len;
        }
        return x;
    }
    pub fn scan_top(&mut self) -> usize {
        assert!((!self.scan_stack_empty));
        return self.scan_stack[self.top];
    }
    pub fn scan_pop_bottom(&mut self) -> usize {
        assert!((!self.scan_stack_empty));
        let x = self.scan_stack[self.bottom];
        if self.top == self.bottom {
            self.scan_stack_empty = true;
        } else {
            self.bottom += 1; self.bottom %= self.buf_len;
        }
        return x;
    }
    pub fn advance_right(&mut self) {
        self.right += 1;
        self.right %= self.buf_len;
        assert!((self.right != self.left));
    }
    pub fn advance_left(&mut self) -> io::Result<()> {
        debug!("advance_left Vec<{},{}>, sizeof({})={}", self.left, self.right,
               self.left, self.size[self.left]);

        let mut left_size = self.size[self.left];

        while left_size >= 0 {
            let left = self.token[self.left].clone();

            let len = match left {
                Token::Break(b) => b.blank_space,
                Token::String(_, len) => {
                    assert_eq!(len, left_size);
                    len
                }
                _ => 0
            };

            try!(self.print(left, left_size));

            self.left_total += len;

            if self.left == self.right {
                break;
            }

            self.left += 1;
            self.left %= self.buf_len;

            left_size = self.size[self.left];
        }

        Ok(())
    }
    pub fn check_stack(&mut self, k: isize) {
        if !self.scan_stack_empty {
            let x = self.scan_top();
            match self.token[x] {
                Token::Begin(_) => {
                    if k > 0 {
                        let popped = self.scan_pop();
                        self.size[popped] = self.size[x] + self.right_total;
                        self.check_stack(k - 1);
                    }
                }
                Token::End => {
                    // paper says + not =, but that makes no sense.
                    let popped = self.scan_pop();
                    self.size[popped] = 1;
                    self.check_stack(k + 1);
                }
                _ => {
                    let popped = self.scan_pop();
                    self.size[popped] = self.size[x] + self.right_total;
                    if k > 0 {
                        self.check_stack(k);
                    }
                }
            }
        }
    }
    pub fn print_newline(&mut self, amount: isize) -> io::Result<()> {
        debug!("NEWLINE {}", amount);
        let ret = write!(self.out, "\n");
        self.pending_indentation = 0;
        self.indent(amount);
        return ret;
    }
    pub fn indent(&mut self, amount: isize) {
        debug!("INDENT {}", amount);
        self.pending_indentation += amount;
    }
    pub fn get_top(&mut self) -> PrintStackElem {
        let print_stack = &mut self.print_stack;
        let n = print_stack.len();
        if n != 0 {
            (*print_stack)[n - 1]
        } else {
            PrintStackElem {
                offset: 0,
                pbreak: PrintStackBreak::Broken(Breaks::Inconsistent)
            }
        }
    }
    pub fn print_str(&mut self, s: &str) -> io::Result<()> {
        while self.pending_indentation > 0 {
            try!(write!(self.out, " "));
            self.pending_indentation -= 1;
        }
        write!(self.out, "{}", s)
    }
    pub fn print(&mut self, token: Token, l: isize) -> io::Result<()> {
        debug!("print {} {} (remaining line space={})", tok_str(&token), l,
               self.space);
        debug!("{}", buf_str(&self.token,
                             &self.size,
                             self.left,
                             self.right,
                             6));
        match token {
          Token::Begin(b) => {
            if l > self.space {
                let col = self.margin - self.space + b.offset;
                debug!("print Begin -> push broken block at col {}", col);
                self.print_stack.push(PrintStackElem {
                    offset: col,
                    pbreak: PrintStackBreak::Broken(b.breaks)
                });
            } else {
                debug!("print Begin -> push fitting block");
                self.print_stack.push(PrintStackElem {
                    offset: 0,
                    pbreak: PrintStackBreak::Fits
                });
            }
            Ok(())
          }
          Token::End => {
            debug!("print End -> pop End");
            let print_stack = &mut self.print_stack;
            assert!((!print_stack.is_empty()));
            print_stack.pop().unwrap();
            Ok(())
          }
          Token::Break(b) => {
            let top = self.get_top();
            match top.pbreak {
              PrintStackBreak::Fits => {
                debug!("print Break({}) in fitting block", b.blank_space);
                self.space -= b.blank_space;
                self.indent(b.blank_space);
                Ok(())
              }
              PrintStackBreak::Broken(Breaks::Consistent) => {
                debug!("print Break({}+{}) in consistent block",
                       top.offset, b.offset);
                let ret = self.print_newline(top.offset + b.offset);
                self.space = self.margin - (top.offset + b.offset);
                ret
              }
              PrintStackBreak::Broken(Breaks::Inconsistent) => {
                if l > self.space {
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
          Token::String(s, len) => {
            debug!("print String({})", s);
            assert_eq!(l, len);
            // assert!(l <= space);
            self.space -= len;
            self.print_str(&s[..])
          }
          Token::Eof => {
            // Eof should never get here.
            panic!();
          }
        }
    }
}

// Convenience functions to talk to the printer.
//
// "raw box"
pub fn rbox(p: &mut Printer, indent: usize, b: Breaks) -> io::Result<()> {
    p.pretty_print(Token::Begin(BeginToken {
        offset: indent as isize,
        breaks: b
    }))
}

pub fn ibox(p: &mut Printer, indent: usize) -> io::Result<()> {
    rbox(p, indent, Breaks::Inconsistent)
}

pub fn cbox(p: &mut Printer, indent: usize) -> io::Result<()> {
    rbox(p, indent, Breaks::Consistent)
}

pub fn break_offset(p: &mut Printer, n: usize, off: isize) -> io::Result<()> {
    p.pretty_print(Token::Break(BreakToken {
        offset: off,
        blank_space: n as isize
    }))
}

pub fn end(p: &mut Printer) -> io::Result<()> {
    p.pretty_print(Token::End)
}

pub fn eof(p: &mut Printer) -> io::Result<()> {
    p.pretty_print(Token::Eof)
}

pub fn word(p: &mut Printer, wrd: &str) -> io::Result<()> {
    p.pretty_print(Token::String(/* bad */ wrd.to_string(), wrd.len() as isize))
}

pub fn huge_word(p: &mut Printer, wrd: &str) -> io::Result<()> {
    p.pretty_print(Token::String(/* bad */ wrd.to_string(), SIZE_INFINITY))
}

pub fn zero_word(p: &mut Printer, wrd: &str) -> io::Result<()> {
    p.pretty_print(Token::String(/* bad */ wrd.to_string(), 0))
}

pub fn spaces(p: &mut Printer, n: usize) -> io::Result<()> {
    break_offset(p, n, 0)
}

pub fn zerobreak(p: &mut Printer) -> io::Result<()> {
    spaces(p, 0)
}

pub fn space(p: &mut Printer) -> io::Result<()> {
    spaces(p, 1)
}

pub fn hardbreak(p: &mut Printer) -> io::Result<()> {
    spaces(p, SIZE_INFINITY as usize)
}

pub fn hardbreak_tok_offset(off: isize) -> Token {
    Token::Break(BreakToken {offset: off, blank_space: SIZE_INFINITY})
}

pub fn hardbreak_tok() -> Token {
    hardbreak_tok_offset(0)
}

//! This pretty-printer is a direct reimplementation of Philip Karlton's
//! Mesa pretty-printer, as described in appendix A of
//!
//! ```text
//! STAN-CS-79-770: "Pretty Printing", by Derek C. Oppen.
//! Stanford Department of Computer Science, 1979.
//! ```
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
//!
//! # Explanation
//!
//! In case you do not have the paper, here is an explanation of what's going
//! on.
//!
//! There is a stream of input tokens flowing through this printer.
//!
//! The printer buffers up to 3N tokens inside itself, where N is linewidth.
//! Yes, linewidth is chars and tokens are multi-char, but in the worst
//! case every token worth buffering is 1 char long, so it's ok.
//!
//! Tokens are String, Break, and Begin/End to delimit blocks.
//!
//! Begin tokens can carry an offset, saying "how far to indent when you break
//! inside here", as well as a flag indicating "consistent" or "inconsistent"
//! breaking. Consistent breaking means that after the first break, no attempt
//! will be made to flow subsequent breaks together onto lines. Inconsistent
//! is the opposite. Inconsistent breaking example would be, say:
//!
//! ```
//! foo(hello, there, good, friends)
//! ```
//!
//! breaking inconsistently to become
//!
//! ```
//! foo(hello, there
//!     good, friends);
//! ```
//!
//! whereas a consistent breaking would yield:
//!
//! ```
//! foo(hello,
//!     there
//!     good,
//!     friends);
//! ```
//!
//! That is, in the consistent-break blocks we value vertical alignment
//! more than the ability to cram stuff onto a line. But in all cases if it
//! can make a block a one-liner, it'll do so.
//!
//! Carrying on with high-level logic:
//!
//! The buffered tokens go through a ring-buffer, 'tokens'. The 'left' and
//! 'right' indices denote the active portion of the ring buffer as well as
//! describing hypothetical points-in-the-infinite-stream at most 3N tokens
//! apart (i.e., "not wrapped to ring-buffer boundaries"). The paper will switch
//! between using 'left' and 'right' terms to denote the wrapped-to-ring-buffer
//! and point-in-infinite-stream senses freely.
//!
//! There is a parallel ring buffer, `size`, that holds the calculated size of
//! each token. Why calculated? Because for Begin/End pairs, the "size"
//! includes everything between the pair. That is, the "size" of Begin is
//! actually the sum of the sizes of everything between Begin and the paired
//! End that follows. Since that is arbitrarily far in the future, `size` is
//! being rewritten regularly while the printer runs; in fact most of the
//! machinery is here to work out `size` entries on the fly (and give up when
//! they're so obviously over-long that "infinity" is a good enough
//! approximation for purposes of line breaking).
//!
//! The "input side" of the printer is managed as an abstract process called
//! SCAN, which uses `scan_stack`, to manage calculating `size`. SCAN is, in
//! other words, the process of calculating 'size' entries.
//!
//! The "output side" of the printer is managed by an abstract process called
//! PRINT, which uses `print_stack`, `margin` and `space` to figure out what to
//! do with each token/size pair it consumes as it goes. It's trying to consume
//! the entire buffered window, but can't output anything until the size is >=
//! 0 (sizes are set to negative while they're pending calculation).
//!
//! So SCAN takes input and buffers tokens and pending calculations, while
//! PRINT gobbles up completed calculations and tokens from the buffer. The
//! theory is that the two can never get more than 3N tokens apart, because
//! once there's "obviously" too much data to fit on a line, in a size
//! calculation, SCAN will write "infinity" to the size and let PRINT consume
//! it.
//!
//! In this implementation (following the paper, again) the SCAN process is the
//! methods called `Printer::pretty_print_*`, and the 'PRINT' process is the
//! method called `Printer::print`.

use std::collections::VecDeque;
use std::fmt;
use std::borrow::Cow;
use log::debug;

/// How to break. Described in more detail in the module docs.
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
    // In practice a string token contains either a `&'static str` or a
    // `String`. `Cow` is overkill for this because we never modify the data,
    // but it's more convenient than rolling our own more specialized type.
    String(Cow<'static, str>, isize),
    Break(BreakToken),
    Begin(BeginToken),
    End,
    Eof,
}

impl Token {
    crate fn is_eof(&self) -> bool {
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

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Token::String(ref s, len) => write!(f, "STR({},{})", s, len),
            Token::Break(_) => f.write_str("BREAK"),
            Token::Begin(_) => f.write_str("BEGIN"),
            Token::End => f.write_str("END"),
            Token::Eof => f.write_str("EOF"),
        }
    }
}

fn buf_str(buf: &[BufEntry], left: usize, right: usize, lim: usize) -> String {
    let n = buf.len();
    let mut i = left;
    let mut l = lim;
    let mut s = String::from("[");
    while i != right && l != 0 {
        l -= 1;
        if i != left {
            s.push_str(", ");
        }
        s.push_str(&format!("{}={}", buf[i].size, &buf[i].token));
        i += 1;
        i %= n;
    }
    s.push(']');
    s
}

#[derive(Copy, Clone)]
crate enum PrintStackBreak {
    Fits,
    Broken(Breaks),
}

#[derive(Copy, Clone)]
crate struct PrintStackElem {
    offset: isize,
    pbreak: PrintStackBreak
}

const SIZE_INFINITY: isize = 0xffff;

pub fn mk_printer(out: &mut String) -> Printer<'_> {
    let linewidth = 78;
    // Yes 55, it makes the ring buffers big enough to never fall behind.
    let n: usize = 55 * linewidth;
    debug!("mk_printer {}", linewidth);
    Printer {
        out,
        buf_max_len: n,
        margin: linewidth as isize,
        space: linewidth as isize,
        left: 0,
        right: 0,
        // Initialize a single entry; advance_right() will extend it on demand
        // up to `buf_max_len` elements.
        buf: vec![BufEntry::default()],
        left_total: 0,
        right_total: 0,
        scan_stack: VecDeque::new(),
        print_stack: Vec::new(),
        pending_indentation: 0
    }
}

pub struct Printer<'a> {
    out: &'a mut String,
    buf_max_len: usize,
    /// Width of lines we're constrained to
    margin: isize,
    /// Number of spaces left on line
    space: isize,
    /// Index of left side of input stream
    left: usize,
    /// Index of right side of input stream
    right: usize,
    /// Ring-buffer of tokens and calculated sizes
    buf: Vec<BufEntry>,
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
    scan_stack: VecDeque<usize>,
    /// Stack of blocks-in-progress being flushed by print
    print_stack: Vec<PrintStackElem> ,
    /// Buffered indentation to avoid writing trailing whitespace
    pending_indentation: isize,
}

#[derive(Clone)]
struct BufEntry {
    token: Token,
    size: isize,
}

impl Default for BufEntry {
    fn default() -> Self {
        BufEntry { token: Token::Eof, size: 0 }
    }
}

impl<'a> Printer<'a> {
    pub fn last_token(&mut self) -> Token {
        self.buf[self.right].token.clone()
    }

    /// Be very careful with this!
    pub fn replace_last_token(&mut self, t: Token) {
        self.buf[self.right].token = t;
    }

    fn pretty_print_eof(&mut self) {
        if !self.scan_stack.is_empty() {
            self.check_stack(0);
            self.advance_left();
        }
        self.indent(0);
    }

    fn pretty_print_begin(&mut self, b: BeginToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.left = 0;
            self.right = 0;
        } else {
            self.advance_right();
        }
        debug!("pp Begin({})/buffer Vec<{},{}>",
               b.offset, self.left, self.right);
        self.buf[self.right] = BufEntry { token: Token::Begin(b), size: -self.right_total };
        let right = self.right;
        self.scan_push(right);
    }

    fn pretty_print_end(&mut self) {
        if self.scan_stack.is_empty() {
            debug!("pp End/print Vec<{},{}>", self.left, self.right);
            self.print_end();
        } else {
            debug!("pp End/buffer Vec<{},{}>", self.left, self.right);
            self.advance_right();
            self.buf[self.right] = BufEntry { token: Token::End, size: -1 };
            let right = self.right;
            self.scan_push(right);
        }
    }

    fn pretty_print_break(&mut self, b: BreakToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.left = 0;
            self.right = 0;
        } else {
            self.advance_right();
        }
        debug!("pp Break({})/buffer Vec<{},{}>",
               b.offset, self.left, self.right);
        self.check_stack(0);
        let right = self.right;
        self.scan_push(right);
        self.buf[self.right] = BufEntry { token: Token::Break(b), size: -self.right_total };
        self.right_total += b.blank_space;
    }

    fn pretty_print_string(&mut self, s: Cow<'static, str>, len: isize) {
        if self.scan_stack.is_empty() {
            debug!("pp String('{}')/print Vec<{},{}>",
                   s, self.left, self.right);
            self.print_string(s, len);
        } else {
            debug!("pp String('{}')/buffer Vec<{},{}>",
                   s, self.left, self.right);
            self.advance_right();
            self.buf[self.right] = BufEntry { token: Token::String(s, len), size: len };
            self.right_total += len;
            self.check_stream();
        }
    }

    crate fn check_stream(&mut self) {
        debug!("check_stream Vec<{}, {}> with left_total={}, right_total={}",
               self.left, self.right, self.left_total, self.right_total);
        if self.right_total - self.left_total > self.space {
            debug!("scan window is {}, longer than space on line ({})",
                   self.right_total - self.left_total, self.space);
            if Some(&self.left) == self.scan_stack.back() {
                debug!("setting {} to infinity and popping", self.left);
                let scanned = self.scan_pop_bottom();
                self.buf[scanned].size = SIZE_INFINITY;
            }
            self.advance_left();
            if self.left != self.right {
                self.check_stream();
            }
        }
    }

    crate fn scan_push(&mut self, x: usize) {
        debug!("scan_push {}", x);
        self.scan_stack.push_front(x);
    }

    crate fn scan_pop(&mut self) -> usize {
        self.scan_stack.pop_front().unwrap()
    }

    crate fn scan_top(&mut self) -> usize {
        *self.scan_stack.front().unwrap()
    }

    crate fn scan_pop_bottom(&mut self) -> usize {
        self.scan_stack.pop_back().unwrap()
    }

    crate fn advance_right(&mut self) {
        self.right += 1;
        self.right %= self.buf_max_len;
        // Extend the buf if necessary.
        if self.right == self.buf.len() {
            self.buf.push(BufEntry::default());
        }
        assert_ne!(self.right, self.left);
    }

    crate fn advance_left(&mut self) {
        debug!("advance_left Vec<{},{}>, sizeof({})={}", self.left, self.right,
               self.left, self.buf[self.left].size);

        let mut left_size = self.buf[self.left].size;

        while left_size >= 0 {
            let left = self.buf[self.left].token.clone();

            let len = match left {
                Token::Break(b) => b.blank_space,
                Token::String(_, len) => {
                    assert_eq!(len, left_size);
                    len
                }
                _ => 0
            };

            self.print(left, left_size);

            self.left_total += len;

            if self.left == self.right {
                break;
            }

            self.left += 1;
            self.left %= self.buf_max_len;

            left_size = self.buf[self.left].size;
        }
    }

    crate fn check_stack(&mut self, k: isize) {
        if !self.scan_stack.is_empty() {
            let x = self.scan_top();
            match self.buf[x].token {
                Token::Begin(_) => {
                    if k > 0 {
                        let popped = self.scan_pop();
                        self.buf[popped].size = self.buf[x].size + self.right_total;
                        self.check_stack(k - 1);
                    }
                }
                Token::End => {
                    // paper says + not =, but that makes no sense.
                    let popped = self.scan_pop();
                    self.buf[popped].size = 1;
                    self.check_stack(k + 1);
                }
                _ => {
                    let popped = self.scan_pop();
                    self.buf[popped].size = self.buf[x].size + self.right_total;
                    if k > 0 {
                        self.check_stack(k);
                    }
                }
            }
        }
    }

    crate fn print_newline(&mut self, amount: isize) {
        debug!("NEWLINE {}", amount);
        self.out.push('\n');
        self.pending_indentation = 0;
        self.indent(amount);
    }

    crate fn indent(&mut self, amount: isize) {
        debug!("INDENT {}", amount);
        self.pending_indentation += amount;
    }

    crate fn get_top(&mut self) -> PrintStackElem {
        match self.print_stack.last() {
            Some(el) => *el,
            None => PrintStackElem {
                offset: 0,
                pbreak: PrintStackBreak::Broken(Breaks::Inconsistent)
            }
        }
    }

    crate fn print_begin(&mut self, b: BeginToken, l: isize) {
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
    }

    crate fn print_end(&mut self) {
        debug!("print End -> pop End");
        let print_stack = &mut self.print_stack;
        assert!(!print_stack.is_empty());
        print_stack.pop().unwrap();
    }

    crate fn print_break(&mut self, b: BreakToken, l: isize) {
        let top = self.get_top();
        match top.pbreak {
            PrintStackBreak::Fits => {
                debug!("print Break({}) in fitting block", b.blank_space);
                self.space -= b.blank_space;
                self.indent(b.blank_space);
            }
            PrintStackBreak::Broken(Breaks::Consistent) => {
                debug!("print Break({}+{}) in consistent block",
                       top.offset, b.offset);
                self.print_newline(top.offset + b.offset);
                self.space = self.margin - (top.offset + b.offset);
            }
            PrintStackBreak::Broken(Breaks::Inconsistent) => {
                if l > self.space {
                    debug!("print Break({}+{}) w/ newline in inconsistent",
                           top.offset, b.offset);
                    self.print_newline(top.offset + b.offset);
                    self.space = self.margin - (top.offset + b.offset);
                } else {
                    debug!("print Break({}) w/o newline in inconsistent",
                           b.blank_space);
                    self.indent(b.blank_space);
                    self.space -= b.blank_space;
                }
            }
        }
    }

    crate fn print_string(&mut self, s: Cow<'static, str>, len: isize) {
        debug!("print String({})", s);
        // assert!(len <= space);
        self.space -= len;

        // Write the pending indent. A more concise way of doing this would be:
        //
        //   write!(self.out, "{: >n$}", "", n = self.pending_indentation as usize)?;
        //
        // But that is significantly slower. This code is sufficiently hot, and indents can get
        // sufficiently large, that the difference is significant on some workloads.
        self.out.reserve(self.pending_indentation as usize);
        self.out.extend(std::iter::repeat(' ').take(self.pending_indentation as usize));
        self.pending_indentation = 0;
        self.out.push_str(&s);
    }

    crate fn print(&mut self, token: Token, l: isize) {
        debug!("print {} {} (remaining line space={})", token, l,
               self.space);
        debug!("{}", buf_str(&self.buf,
                             self.left,
                             self.right,
                             6));
        match token {
            Token::Begin(b) => self.print_begin(b, l),
            Token::End => self.print_end(),
            Token::Break(b) => self.print_break(b, l),
            Token::String(s, len) => {
                assert_eq!(len, l);
                self.print_string(s, len);
            }
            Token::Eof => panic!(), // Eof should never get here.
        }
    }

    // Convenience functions to talk to the printer.

    /// "raw box"
    crate fn rbox(&mut self, indent: usize, b: Breaks) {
        self.pretty_print_begin(BeginToken {
            offset: indent as isize,
            breaks: b
        })
    }

    /// Inconsistent breaking box
    crate fn ibox(&mut self, indent: usize) {
        self.rbox(indent, Breaks::Inconsistent)
    }

    /// Consistent breaking box
    pub fn cbox(&mut self, indent: usize) {
        self.rbox(indent, Breaks::Consistent)
    }

    pub fn break_offset(&mut self, n: usize, off: isize) {
        self.pretty_print_break(BreakToken {
            offset: off,
            blank_space: n as isize
        })
    }

    crate fn end(&mut self) {
        self.pretty_print_end()
    }

    pub fn eof(&mut self) {
        self.pretty_print_eof()
    }

    pub fn word<S: Into<Cow<'static, str>>>(&mut self, wrd: S) {
        let s = wrd.into();
        let len = s.len() as isize;
        self.pretty_print_string(s, len)
    }

    fn spaces(&mut self, n: usize) {
        self.break_offset(n, 0)
    }

    crate fn zerobreak(&mut self) {
        self.spaces(0)
    }

    pub fn space(&mut self) {
        self.spaces(1)
    }

    pub fn hardbreak(&mut self) {
        self.spaces(SIZE_INFINITY as usize)
    }

    pub fn hardbreak_tok_offset(off: isize) -> Token {
        Token::Break(BreakToken {offset: off, blank_space: SIZE_INFINITY})
    }
}

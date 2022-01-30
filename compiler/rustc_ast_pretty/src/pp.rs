//! This pretty-printer is a direct reimplementation of Philip Karlton's
//! Mesa pretty-printer, as described in the appendix to
//! Derek C. Oppen, "Pretty Printing" (1979),
//! Stanford Computer Science Department STAN-CS-79-770,
//! <http://i.stanford.edu/pub/cstr/reports/cs/tr/79/770/CS-TR-79-770.pdf>.
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
//! foo(hello, there,
//!     good, friends);
//! ```
//!
//! whereas a consistent breaking would yield:
//!
//! ```
//! foo(hello,
//!     there,
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
//! methods called `Printer::scan_*`, and the 'PRINT' process is the
//! method called `Printer::print`.

mod ring;

use ring::RingBuffer;
use std::borrow::Cow;
use std::collections::VecDeque;
use std::iter;

/// How to break. Described in more detail in the module docs.
#[derive(Clone, Copy, PartialEq)]
pub enum Breaks {
    Consistent,
    Inconsistent,
}

#[derive(Clone, Copy)]
pub struct BreakToken {
    offset: isize,
    blank_space: isize,
}

#[derive(Clone, Copy)]
pub struct BeginToken {
    offset: isize,
    breaks: Breaks,
}

#[derive(Clone)]
pub enum Token {
    // In practice a string token contains either a `&'static str` or a
    // `String`. `Cow` is overkill for this because we never modify the data,
    // but it's more convenient than rolling our own more specialized type.
    String(Cow<'static, str>),
    Break(BreakToken),
    Begin(BeginToken),
    End,
}

impl Token {
    pub fn is_hardbreak_tok(&self) -> bool {
        matches!(self, Token::Break(BreakToken { offset: 0, blank_space: SIZE_INFINITY }))
    }
}

#[derive(Copy, Clone)]
enum PrintFrame {
    Fits,
    Broken { offset: isize, breaks: Breaks },
}

const SIZE_INFINITY: isize = 0xffff;

pub struct Printer {
    out: String,
    /// Width of lines we're constrained to
    margin: isize,
    /// Number of spaces left on line
    space: isize,
    /// Ring-buffer of tokens and calculated sizes
    buf: RingBuffer<BufEntry>,
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
    print_stack: Vec<PrintFrame>,
    /// Buffered indentation to avoid writing trailing whitespace
    pending_indentation: isize,
    /// The token most recently popped from the left boundary of the
    /// ring-buffer for printing
    last_printed: Option<Token>,
}

#[derive(Clone)]
struct BufEntry {
    token: Token,
    size: isize,
}

impl Printer {
    pub fn new() -> Self {
        let linewidth = 78;
        Printer {
            out: String::new(),
            margin: linewidth as isize,
            space: linewidth as isize,
            buf: RingBuffer::new(),
            left_total: 0,
            right_total: 0,
            scan_stack: VecDeque::new(),
            print_stack: Vec::new(),
            pending_indentation: 0,
            last_printed: None,
        }
    }

    pub fn last_token(&self) -> Option<&Token> {
        self.last_token_still_buffered().or_else(|| self.last_printed.as_ref())
    }

    pub fn last_token_still_buffered(&self) -> Option<&Token> {
        self.buf.last().map(|last| &last.token)
    }

    /// Be very careful with this!
    pub fn replace_last_token_still_buffered(&mut self, token: Token) {
        self.buf.last_mut().unwrap().token = token;
    }

    fn scan_eof(&mut self) {
        if !self.scan_stack.is_empty() {
            self.check_stack(0);
            self.advance_left();
        }
    }

    fn scan_begin(&mut self, token: BeginToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.buf.clear();
        }
        let right = self.buf.push(BufEntry { token: Token::Begin(token), size: -self.right_total });
        self.scan_stack.push_back(right);
    }

    fn scan_end(&mut self) {
        if self.scan_stack.is_empty() {
            self.print_end();
        } else {
            let right = self.buf.push(BufEntry { token: Token::End, size: -1 });
            self.scan_stack.push_back(right);
        }
    }

    fn scan_break(&mut self, token: BreakToken) {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.buf.clear();
        } else {
            self.check_stack(0);
        }
        let right = self.buf.push(BufEntry { token: Token::Break(token), size: -self.right_total });
        self.scan_stack.push_back(right);
        self.right_total += token.blank_space;
    }

    fn scan_string(&mut self, string: Cow<'static, str>) {
        if self.scan_stack.is_empty() {
            self.print_string(&string);
        } else {
            let len = string.len() as isize;
            self.buf.push(BufEntry { token: Token::String(string), size: len });
            self.right_total += len;
            self.check_stream();
        }
    }

    fn check_stream(&mut self) {
        while self.right_total - self.left_total > self.space {
            if *self.scan_stack.front().unwrap() == self.buf.index_of_first() {
                self.scan_stack.pop_front().unwrap();
                self.buf.first_mut().unwrap().size = SIZE_INFINITY;
            }
            self.advance_left();
            if self.buf.is_empty() {
                break;
            }
        }
    }

    fn advance_left(&mut self) {
        while self.buf.first().unwrap().size >= 0 {
            let left = self.buf.pop_first().unwrap();

            match &left.token {
                Token::String(string) => {
                    self.left_total += string.len() as isize;
                    self.print_string(string);
                }
                Token::Break(token) => {
                    self.left_total += token.blank_space;
                    self.print_break(*token, left.size);
                }
                Token::Begin(token) => self.print_begin(*token, left.size),
                Token::End => self.print_end(),
            }

            self.last_printed = Some(left.token);

            if self.buf.is_empty() {
                break;
            }
        }
    }

    fn check_stack(&mut self, mut depth: usize) {
        while let Some(&index) = self.scan_stack.back() {
            let mut entry = &mut self.buf[index];
            match entry.token {
                Token::Begin(_) => {
                    if depth == 0 {
                        break;
                    }
                    self.scan_stack.pop_back().unwrap();
                    entry.size += self.right_total;
                    depth -= 1;
                }
                Token::End => {
                    // paper says + not =, but that makes no sense.
                    self.scan_stack.pop_back().unwrap();
                    entry.size = 1;
                    depth += 1;
                }
                _ => {
                    self.scan_stack.pop_back().unwrap();
                    entry.size += self.right_total;
                    if depth == 0 {
                        break;
                    }
                }
            }
        }
    }

    fn get_top(&self) -> PrintFrame {
        *self
            .print_stack
            .last()
            .unwrap_or(&PrintFrame::Broken { offset: 0, breaks: Breaks::Inconsistent })
    }

    fn print_begin(&mut self, token: BeginToken, size: isize) {
        if size > self.space {
            let col = self.margin - self.space + token.offset;
            self.print_stack.push(PrintFrame::Broken { offset: col, breaks: token.breaks });
        } else {
            self.print_stack.push(PrintFrame::Fits);
        }
    }

    fn print_end(&mut self) {
        self.print_stack.pop().unwrap();
    }

    fn print_break(&mut self, token: BreakToken, size: isize) {
        let break_offset =
            match self.get_top() {
                PrintFrame::Fits => None,
                PrintFrame::Broken { offset, breaks: Breaks::Consistent } => Some(offset),
                PrintFrame::Broken { offset, breaks: Breaks::Inconsistent } => {
                    if size > self.space { Some(offset) } else { None }
                }
            };
        if let Some(offset) = break_offset {
            self.out.push('\n');
            self.pending_indentation = offset + token.offset;
            self.space = self.margin - (offset + token.offset);
        } else {
            self.pending_indentation += token.blank_space;
            self.space -= token.blank_space;
        }
    }

    fn print_string(&mut self, string: &str) {
        // Write the pending indent. A more concise way of doing this would be:
        //
        //   write!(self.out, "{: >n$}", "", n = self.pending_indentation as usize)?;
        //
        // But that is significantly slower. This code is sufficiently hot, and indents can get
        // sufficiently large, that the difference is significant on some workloads.
        self.out.reserve(self.pending_indentation as usize);
        self.out.extend(iter::repeat(' ').take(self.pending_indentation as usize));
        self.pending_indentation = 0;

        self.out.push_str(string);
        self.space -= string.len() as isize;
    }

    // Convenience functions to talk to the printer.

    /// "raw box"
    pub fn rbox(&mut self, indent: usize, breaks: Breaks) {
        self.scan_begin(BeginToken { offset: indent as isize, breaks })
    }

    /// Inconsistent breaking box
    pub fn ibox(&mut self, indent: usize) {
        self.rbox(indent, Breaks::Inconsistent)
    }

    /// Consistent breaking box
    pub fn cbox(&mut self, indent: usize) {
        self.rbox(indent, Breaks::Consistent)
    }

    pub fn break_offset(&mut self, n: usize, off: isize) {
        self.scan_break(BreakToken { offset: off, blank_space: n as isize })
    }

    pub fn end(&mut self) {
        self.scan_end()
    }

    pub fn eof(mut self) -> String {
        self.scan_eof();
        self.out
    }

    pub fn word<S: Into<Cow<'static, str>>>(&mut self, wrd: S) {
        let string = wrd.into();
        self.scan_string(string)
    }

    fn spaces(&mut self, n: usize) {
        self.break_offset(n, 0)
    }

    pub fn zerobreak(&mut self) {
        self.spaces(0)
    }

    pub fn space(&mut self) {
        self.spaces(1)
    }

    pub fn hardbreak(&mut self) {
        self.spaces(SIZE_INFINITY as usize)
    }

    pub fn is_beginning_of_line(&self) -> bool {
        match self.last_token() {
            Some(last_token) => last_token.is_hardbreak_tok(),
            None => true,
        }
    }

    pub fn hardbreak_tok_offset(off: isize) -> Token {
        Token::Break(BreakToken { offset: off, blank_space: SIZE_INFINITY })
    }
}

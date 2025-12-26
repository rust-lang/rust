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
//! breaking.
//!
//! The buffered tokens go through a ring-buffer, 'tokens'. The 'left' and
//! 'right' indices denote the active portion of the ring buffer as well as
//! describing hypothetical points-in-the-infinite-stream at most 3N tokens
//! apart.
//!
//! There is a parallel ring buffer, `size`, that holds the calculated size of
//! each token. For Begin/End pairs, the "size" includes everything between the
//! pair.
//!
//! The "input side" of the printer is managed as an abstract process called
//! SCAN.
//!
//! The "output side" of the printer is managed by an abstract process called
//! PRINT.
//!
//! In this implementation the SCAN process is the `scan_*` methods, and PRINT
//! is `print`.

mod convenience;
mod ring;

use std::borrow::Cow;
use std::collections::VecDeque;
use std::{cmp, iter};

use ring::RingBuffer;

/// How to break.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Breaks {
    Consistent,
    Inconsistent,
}

/// Indentation style.
#[derive(Clone, Copy, Debug, PartialEq)]
enum IndentStyle {
    /// Vertically aligned under whatever column this block begins at.
    Visual,
    /// Indented relative to the indentation level of the previous line.
    Block { offset: isize },
}

/// Break token information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BreakToken {
    offset: isize,
    blank_space: isize,
    pre_break: Option<char>,
}

/// Begin token information.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct BeginToken {
    indent: IndentStyle,
    breaks: Breaks,
}

/// Tokens for the printer.
#[derive(PartialEq, Debug)]
pub(crate) enum Token {
    String(Cow<'static, str>),
    Break(BreakToken),
    Begin(BeginToken),
    End,
}

/// Frame used for printing.
#[derive(Copy, Clone, Debug)]
enum PrintFrame {
    Fits,
    Broken { indent: usize, breaks: Breaks },
}

/// Line width constants.
const MARGIN: isize = 78;
const MIN_SPACE: isize = 60;
const SIZE_INFINITY: isize = isize::MAX / 2;

/// Ring-buffer entry.
struct BufEntry {
    token: Token,
    size: isize,
}

/// Pretty-printer.
pub struct Printer {
    out: String,
    space: isize,
    buf: RingBuffer<BufEntry>,
    left_total: isize,
    right_total: isize,
    scan_stack: VecDeque<usize>,
    print_stack: Vec<PrintFrame>,
    indent: usize,
    pending_indentation: isize,
    last_printed: Option<Token>,
}

/// Marker used to ensure `Begin`/`End` balance.
pub struct BoxMarker {
    valid: bool,
}

impl Drop for BoxMarker {
    fn drop(&mut self) {
        if self.valid {
            panic!("BoxMarker not ended with `Printer::scan_end()`");
        }
    }
}

impl Printer {
    pub fn new() -> Self {
        Printer {
            out: String::new(),
            space: MARGIN,
            buf: RingBuffer::new(),
            left_total: 0,
            right_total: 0,
            scan_stack: VecDeque::new(),
            print_stack: Vec::new(),
            indent: 0,
            pending_indentation: 0,
            last_printed: None,
        }
    }

    /// Flush remaining tokens at EOF.
    pub fn scan_eof(&mut self) {
        self.check_stack();
        while !self.buf.is_empty() {
            self.advance_left();
        }
    }

    /// Begin a block.
    pub fn scan_begin(&mut self, token: BeginToken) -> BoxMarker {
        if self.scan_stack.is_empty() {
            self.left_total = 1;
            self.right_total = 1;
            self.buf.clear();
        }

        let right = self.buf.push(BufEntry {
            token: Token::Begin(token),
            size: -self.right_total,
        });

        self.scan_stack.push_back(right);
        BoxMarker { valid: true }
    }

    /// End a block.
    pub fn scan_end(&mut self, mut marker: BoxMarker) {
        let right = self.buf.push(BufEntry {
            token: Token::End,
            size: -1,
        });
        self.scan_stack.push_back(right);
        marker.valid = false; // mark as used
    }

    /// Advance left-side token processing.
    fn advance_left(&mut self) {
        while let Some(entry) = self.buf.first() {
            if entry.size < 0 {
                break;
            }

            let left = self.buf.pop_first().unwrap();
            match &left.token {
                Token::String(s) => {
                    self.left_total += s.len() as isize;
                    self.print_string(s);
                }
                Token::Break(b) => {
                    self.left_total += b.blank_space;
                    self.print_break(*b);
                }
                Token::Begin(b) => self.print_begin(*b),
                Token::End => self.print_end(),
            }
            self.last_printed = Some(left.token);
        }
    }

    fn print_end(&mut self) {
        if let Some(frame) = self.print_stack.pop() {
            if let PrintFrame::Broken { indent, .. } = frame {
                self.indent = indent;
            }
        } else {
            panic!("print_end called with empty print_stack");
        }
    }

    fn print_string(&mut self, string: &Cow<'static, str>) {
        self.out.reserve(self.pending_indentation as usize);
        self.out
            .extend(iter::repeat(' ').take(self.pending_indentation as usize));
        self.pending_indentation = 0;

        self.out.push_str(string);
        self.space -= string.len() as isize;
        if self.space < 0 {
            self.space = 0;
        }
    }

    fn print_break(&mut self, b: BreakToken) {
        if self.space < 0 {
            self.out.push('\n');
            self.indent = (self.indent as isize + b.offset).max(0) as usize;
            self.pending_indentation = self.indent as isize;
            self.space = MARGIN - self.pending_indentation;
        } else {
            self.out.extend(iter::repeat(' ').take(b.blank_space as usize));
            self.space -= b.blank_space;
        }
    }

    fn print_begin(&mut self, b: BeginToken) {
        self.print_stack.push(PrintFrame::Fits);
        if let IndentStyle::Block { offset } = b.indent {
            self.indent = (self.indent as isize + offset).max(0) as usize;
        }
    }

    fn check_stack(&self) {
        if !self.scan_stack.is_empty() {
            panic!("Unbalanced begin/end tokens detected");
        }
    }

    pub fn output(&self) -> &str {
        &self.out
    }
}

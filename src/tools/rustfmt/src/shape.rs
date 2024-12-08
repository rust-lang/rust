use std::borrow::Cow;
use std::cmp::min;
use std::ops::{Add, Sub};

use crate::Config;

#[derive(Copy, Clone, Debug)]
pub(crate) struct Indent {
    // Width of the block indent, in characters. Must be a multiple of
    // Config::tab_spaces.
    pub(crate) block_indent: usize,
    // Alignment in characters.
    pub(crate) alignment: usize,
}

// INDENT_BUFFER.len() = 81
const INDENT_BUFFER_LEN: usize = 80;
const INDENT_BUFFER: &str =
    "\n                                                                                ";

impl Indent {
    pub(crate) fn new(block_indent: usize, alignment: usize) -> Indent {
        Indent {
            block_indent,
            alignment,
        }
    }

    pub(crate) fn from_width(config: &Config, width: usize) -> Indent {
        if config.hard_tabs() {
            let tab_num = width / config.tab_spaces();
            let alignment = width % config.tab_spaces();
            Indent::new(config.tab_spaces() * tab_num, alignment)
        } else {
            Indent::new(width, 0)
        }
    }

    pub(crate) fn empty() -> Indent {
        Indent::new(0, 0)
    }

    pub(crate) fn block_only(&self) -> Indent {
        Indent {
            block_indent: self.block_indent,
            alignment: 0,
        }
    }

    pub(crate) fn block_indent(mut self, config: &Config) -> Indent {
        self.block_indent += config.tab_spaces();
        self
    }

    pub(crate) fn block_unindent(mut self, config: &Config) -> Indent {
        if self.block_indent < config.tab_spaces() {
            Indent::new(self.block_indent, 0)
        } else {
            self.block_indent -= config.tab_spaces();
            self
        }
    }

    pub(crate) fn width(&self) -> usize {
        self.block_indent + self.alignment
    }

    pub(crate) fn to_string(&self, config: &Config) -> Cow<'static, str> {
        self.to_string_inner(config, 1)
    }

    pub(crate) fn to_string_with_newline(&self, config: &Config) -> Cow<'static, str> {
        self.to_string_inner(config, 0)
    }

    fn to_string_inner(&self, config: &Config, offset: usize) -> Cow<'static, str> {
        let (num_tabs, num_spaces) = if config.hard_tabs() {
            (self.block_indent / config.tab_spaces(), self.alignment)
        } else {
            (0, self.width())
        };
        let num_chars = num_tabs + num_spaces;
        if num_tabs == 0 && num_chars + offset <= INDENT_BUFFER_LEN {
            Cow::from(&INDENT_BUFFER[offset..=num_chars])
        } else {
            let mut indent = String::with_capacity(num_chars + if offset == 0 { 1 } else { 0 });
            if offset == 0 {
                indent.push('\n');
            }
            for _ in 0..num_tabs {
                indent.push('\t')
            }
            for _ in 0..num_spaces {
                indent.push(' ')
            }
            Cow::from(indent)
        }
    }
}

impl Add for Indent {
    type Output = Indent;

    fn add(self, rhs: Indent) -> Indent {
        Indent {
            block_indent: self.block_indent + rhs.block_indent,
            alignment: self.alignment + rhs.alignment,
        }
    }
}

impl Sub for Indent {
    type Output = Indent;

    fn sub(self, rhs: Indent) -> Indent {
        Indent::new(
            self.block_indent - rhs.block_indent,
            self.alignment - rhs.alignment,
        )
    }
}

impl Add<usize> for Indent {
    type Output = Indent;

    fn add(self, rhs: usize) -> Indent {
        Indent::new(self.block_indent, self.alignment + rhs)
    }
}

impl Sub<usize> for Indent {
    type Output = Indent;

    fn sub(self, rhs: usize) -> Indent {
        Indent::new(self.block_indent, self.alignment - rhs)
    }
}

// 8096 is close enough to infinite for rustfmt.
const INFINITE_SHAPE_WIDTH: usize = 8096;

#[derive(Copy, Clone, Debug)]
pub(crate) struct Shape {
    pub(crate) width: usize,
    // The current indentation of code.
    pub(crate) indent: Indent,
    // Indentation + any already emitted text on the first line of the current
    // statement.
    pub(crate) offset: usize,
}

impl Shape {
    /// `indent` is the indentation of the first line. The next lines
    /// should begin with at least `indent` spaces (except backwards
    /// indentation). The first line should not begin with indentation.
    /// `width` is the maximum number of characters on the last line
    /// (excluding `indent`). The width of other lines is not limited by
    /// `width`.
    /// Note that in reality, we sometimes use width for lines other than the
    /// last (i.e., we are conservative).
    // .......*-------*
    //        |       |
    //        |     *-*
    //        *-----|
    // |<------------>|  max width
    // |<---->|          indent
    //        |<--->|    width
    pub(crate) fn legacy(width: usize, indent: Indent) -> Shape {
        Shape {
            width,
            indent,
            offset: indent.alignment,
        }
    }

    pub(crate) fn indented(indent: Indent, config: &Config) -> Shape {
        Shape {
            width: config.max_width().saturating_sub(indent.width()),
            indent,
            offset: indent.alignment,
        }
    }

    pub(crate) fn with_max_width(&self, config: &Config) -> Shape {
        Shape {
            width: config.max_width().saturating_sub(self.indent.width()),
            ..*self
        }
    }

    pub(crate) fn visual_indent(&self, extra_width: usize) -> Shape {
        let alignment = self.offset + extra_width;
        Shape {
            width: self.width,
            indent: Indent::new(self.indent.block_indent, alignment),
            offset: alignment,
        }
    }

    pub(crate) fn block_indent(&self, extra_width: usize) -> Shape {
        if self.indent.alignment == 0 {
            Shape {
                width: self.width,
                indent: Indent::new(self.indent.block_indent + extra_width, 0),
                offset: 0,
            }
        } else {
            Shape {
                width: self.width,
                indent: self.indent + extra_width,
                offset: self.indent.alignment + extra_width,
            }
        }
    }

    pub(crate) fn block_left(&self, width: usize) -> Option<Shape> {
        self.block_indent(width).sub_width(width)
    }

    pub(crate) fn add_offset(&self, extra_width: usize) -> Shape {
        Shape {
            offset: self.offset + extra_width,
            ..*self
        }
    }

    pub(crate) fn block(&self) -> Shape {
        Shape {
            indent: self.indent.block_only(),
            ..*self
        }
    }

    pub(crate) fn saturating_sub_width(&self, width: usize) -> Shape {
        self.sub_width(width).unwrap_or(Shape { width: 0, ..*self })
    }

    pub(crate) fn sub_width(&self, width: usize) -> Option<Shape> {
        Some(Shape {
            width: self.width.checked_sub(width)?,
            ..*self
        })
    }

    pub(crate) fn shrink_left(&self, width: usize) -> Option<Shape> {
        Some(Shape {
            width: self.width.checked_sub(width)?,
            indent: self.indent + width,
            offset: self.offset + width,
        })
    }

    pub(crate) fn offset_left(&self, width: usize) -> Option<Shape> {
        self.add_offset(width).sub_width(width)
    }

    pub(crate) fn used_width(&self) -> usize {
        self.indent.block_indent + self.offset
    }

    pub(crate) fn rhs_overhead(&self, config: &Config) -> usize {
        config
            .max_width()
            .saturating_sub(self.used_width() + self.width)
    }

    pub(crate) fn comment(&self, config: &Config) -> Shape {
        let width = min(
            self.width,
            config.comment_width().saturating_sub(self.indent.width()),
        );
        Shape { width, ..*self }
    }

    pub(crate) fn to_string_with_newline(&self, config: &Config) -> Cow<'static, str> {
        let mut offset_indent = self.indent;
        offset_indent.alignment = self.offset;
        offset_indent.to_string_inner(config, 0)
    }

    /// Creates a `Shape` with a virtually infinite width.
    pub(crate) fn infinite_width(&self) -> Shape {
        Shape {
            width: INFINITE_SHAPE_WIDTH,
            ..*self
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn indent_add_sub() {
        let indent = Indent::new(4, 8) + Indent::new(8, 12);
        assert_eq!(12, indent.block_indent);
        assert_eq!(20, indent.alignment);

        let indent = indent - Indent::new(4, 4);
        assert_eq!(8, indent.block_indent);
        assert_eq!(16, indent.alignment);
    }

    #[test]
    fn indent_add_sub_alignment() {
        let indent = Indent::new(4, 8) + 4;
        assert_eq!(4, indent.block_indent);
        assert_eq!(12, indent.alignment);

        let indent = indent - 4;
        assert_eq!(4, indent.block_indent);
        assert_eq!(8, indent.alignment);
    }

    #[test]
    fn indent_to_string_spaces() {
        let config = Config::default();
        let indent = Indent::new(4, 8);

        // 12 spaces
        assert_eq!("            ", indent.to_string(&config));
    }

    #[test]
    fn indent_to_string_hard_tabs() {
        let mut config = Config::default();
        config.set().hard_tabs(true);
        let indent = Indent::new(8, 4);

        // 2 tabs + 4 spaces
        assert_eq!("\t\t    ", indent.to_string(&config));
    }

    #[test]
    fn shape_visual_indent() {
        let config = Config::default();
        let indent = Indent::new(4, 8);
        let shape = Shape::legacy(config.max_width(), indent);
        let shape = shape.visual_indent(20);

        assert_eq!(config.max_width(), shape.width);
        assert_eq!(4, shape.indent.block_indent);
        assert_eq!(28, shape.indent.alignment);
        assert_eq!(28, shape.offset);
    }

    #[test]
    fn shape_block_indent_without_alignment() {
        let config = Config::default();
        let indent = Indent::new(4, 0);
        let shape = Shape::legacy(config.max_width(), indent);
        let shape = shape.block_indent(20);

        assert_eq!(config.max_width(), shape.width);
        assert_eq!(24, shape.indent.block_indent);
        assert_eq!(0, shape.indent.alignment);
        assert_eq!(0, shape.offset);
    }

    #[test]
    fn shape_block_indent_with_alignment() {
        let config = Config::default();
        let indent = Indent::new(4, 8);
        let shape = Shape::legacy(config.max_width(), indent);
        let shape = shape.block_indent(20);

        assert_eq!(config.max_width(), shape.width);
        assert_eq!(4, shape.indent.block_indent);
        assert_eq!(28, shape.indent.alignment);
        assert_eq!(28, shape.offset);
    }
}

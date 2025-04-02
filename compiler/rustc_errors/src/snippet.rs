// Code for annotating snippets.

use rustc_macros::{Decodable, Encodable};

use crate::{Level, Loc};

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct Line {
    pub line_index: usize,
    pub annotations: Vec<Annotation>,
}

#[derive(Clone, Copy, Debug, PartialOrd, Ord, PartialEq, Eq, Default)]
pub(crate) struct AnnotationColumn {
    /// the (0-indexed) column for *display* purposes, counted in characters, not utf-8 bytes
    pub display: usize,
    /// the (0-indexed) column in the file, counted in characters, not utf-8 bytes.
    ///
    /// this may be different from `self.display`,
    /// e.g. if the file contains hard tabs, because we convert tabs to spaces for error messages.
    ///
    /// for example:
    /// ```text
    /// (hard tab)hello
    ///           ^ this is display column 4, but file column 1
    /// ```
    ///
    /// we want to keep around the correct file offset so that column numbers in error messages
    /// are correct. (motivated by <https://github.com/rust-lang/rust/issues/109537>)
    pub file: usize,
}

impl AnnotationColumn {
    pub(crate) fn from_loc(loc: &Loc) -> AnnotationColumn {
        AnnotationColumn { display: loc.col_display, file: loc.col.0 }
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct MultilineAnnotation {
    pub depth: usize,
    pub line_start: usize,
    pub line_end: usize,
    pub start_col: AnnotationColumn,
    pub end_col: AnnotationColumn,
    pub is_primary: bool,
    pub label: Option<String>,
    pub overlaps_exactly: bool,
}

impl MultilineAnnotation {
    pub(crate) fn increase_depth(&mut self) {
        self.depth += 1;
    }

    /// Compare two `MultilineAnnotation`s considering only the `Span` they cover.
    pub(crate) fn same_span(&self, other: &MultilineAnnotation) -> bool {
        self.line_start == other.line_start
            && self.line_end == other.line_end
            && self.start_col == other.start_col
            && self.end_col == other.end_col
    }

    pub(crate) fn as_start(&self) -> Annotation {
        Annotation {
            start_col: self.start_col,
            end_col: AnnotationColumn {
                // these might not correspond to the same place anymore,
                // but that's okay for our purposes
                display: self.start_col.display + 1,
                file: self.start_col.file + 1,
            },
            is_primary: self.is_primary,
            label: None,
            annotation_type: AnnotationType::MultilineStart(self.depth),
        }
    }

    pub(crate) fn as_end(&self) -> Annotation {
        Annotation {
            start_col: AnnotationColumn {
                // these might not correspond to the same place anymore,
                // but that's okay for our purposes
                display: self.end_col.display.saturating_sub(1),
                file: self.end_col.file.saturating_sub(1),
            },
            end_col: self.end_col,
            is_primary: self.is_primary,
            label: self.label.clone(),
            annotation_type: AnnotationType::MultilineEnd(self.depth),
        }
    }

    pub(crate) fn as_line(&self) -> Annotation {
        Annotation {
            start_col: Default::default(),
            end_col: Default::default(),
            is_primary: self.is_primary,
            label: None,
            annotation_type: AnnotationType::MultilineLine(self.depth),
        }
    }
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) enum AnnotationType {
    /// Annotation under a single line of code
    Singleline,

    // The Multiline type above is replaced with the following three in order
    // to reuse the current label drawing code.
    //
    // Each of these corresponds to one part of the following diagram:
    //
    //     x |   foo(1 + bar(x,
    //       |  _________^              < MultilineStart
    //     x | |             y),        < MultilineLine
    //       | |______________^ label   < MultilineEnd
    //     x |       z);
    /// Annotation marking the first character of a fully shown multiline span
    MultilineStart(usize),
    /// Annotation marking the last character of a fully shown multiline span
    MultilineEnd(usize),
    /// Line at the left enclosing the lines of a fully shown multiline span
    // Just a placeholder for the drawing algorithm, to know that it shouldn't skip the first 4
    // and last 2 lines of code. The actual line is drawn in `emit_message_default` and not in
    // `draw_multiline_line`.
    MultilineLine(usize),
}

#[derive(Clone, Debug, PartialOrd, Ord, PartialEq, Eq)]
pub(crate) struct Annotation {
    /// Start column.
    /// Note that it is important that this field goes
    /// first, so that when we sort, we sort orderings by start
    /// column.
    pub start_col: AnnotationColumn,

    /// End column within the line (exclusive)
    pub end_col: AnnotationColumn,

    /// Is this annotation derived from primary span
    pub is_primary: bool,

    /// Optional label to display adjacent to the annotation.
    pub label: Option<String>,

    /// Is this a single line, multiline or multiline span minimized down to a
    /// smaller span.
    pub annotation_type: AnnotationType,
}

impl Annotation {
    /// Whether this annotation is a vertical line placeholder.
    pub(crate) fn is_line(&self) -> bool {
        matches!(self.annotation_type, AnnotationType::MultilineLine(_))
    }

    /// Length of this annotation as displayed in the stderr output
    pub(crate) fn len(&self) -> usize {
        // Account for usize underflows
        self.end_col.display.abs_diff(self.start_col.display)
    }

    pub(crate) fn has_label(&self) -> bool {
        if let Some(ref label) = self.label {
            // Consider labels with no text as effectively not being there
            // to avoid weird output with unnecessary vertical lines, like:
            //
            //     X | fn foo(x: u32) {
            //       | -------^------
            //       | |      |
            //       | |
            //       |
            //
            // Note that this would be the complete output users would see.
            !label.is_empty()
        } else {
            false
        }
    }

    pub(crate) fn takes_space(&self) -> bool {
        // Multiline annotations always have to keep vertical space.
        matches!(
            self.annotation_type,
            AnnotationType::MultilineStart(_) | AnnotationType::MultilineEnd(_)
        )
    }
}

#[derive(Debug)]
pub(crate) struct StyledString {
    pub text: String,
    pub style: Style,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Encodable, Decodable)]
pub enum Style {
    MainHeaderMsg,
    HeaderMsg,
    LineAndColumn,
    LineNumber,
    Quotation,
    UnderlinePrimary,
    UnderlineSecondary,
    LabelPrimary,
    LabelSecondary,
    NoStyle,
    Level(Level),
    Highlight,
    Addition,
    Removal,
}

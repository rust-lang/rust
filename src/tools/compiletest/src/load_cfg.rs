#![allow(unused)]

mod itemlist;
mod prepare;

use error::{Error, ErrorExt, Result};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LineCol {
    pub line: usize,
    pub col: usize,
}

impl LineCol {
    const fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

/// When we need to e.g. build regexes that include a pattern, we need to know what kind of
/// comments to use. Usually we just build a regex for all expressions, even though we don't
/// use them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CommentTy {
    Slashes,
    Hash,
    Semi,
}

impl CommentTy {
    const fn as_str(self) -> &'static str {
        match self {
            CommentTy::Slashes => "//",
            CommentTy::Hash => "#",
            CommentTy::Semi => ";",
        }
    }

    const fn directive(self) -> &'static str {
        match self {
            CommentTy::Slashes => "//@",
            CommentTy::Hash => "#@",
            CommentTy::Semi => ";@",
        }
    }

    const fn all() -> &'static [Self] {
        &[Self::Slashes, Self::Hash, Self::Semi]
    }
}

/// Errors used within the `load_cfg` module
mod error {
    use std::fmt;
    use std::path::{Path, PathBuf};

    use super::LineCol;

    pub type Error = Box<dyn std::error::Error>;
    pub type Result<T, E = Error> = std::result::Result<T, E>;

    #[derive(Debug)]
    struct FullError {
        msg: Box<str>,
        fname: Option<PathBuf>,
        pos: Option<LineCol>,
        context: Vec<Box<str>>,
    }

    impl fmt::Display for FullError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "error: {}", self.msg)?;

            let path = self.fname.as_ref().map_or(Path::new("unknown").display(), |p| p.display());
            write!(f, "\n parsing '{path}'",)?;

            let pos = self.pos.unwrap_or_default();
            write!(f, " at line {}, column {}", pos.line, pos.col)?;

            if !self.context.is_empty() {
                write!(f, "\ncontext: {:#?}", self.context)?;
            }

            Ok(())
        }
    }

    impl std::error::Error for FullError {}

    /// Give us an easy way to tack context onto an error.
    pub trait ErrorExt {
        fn pos(self, pos: LineCol) -> Self;
        fn line(self, line: usize) -> Self;
        fn col(self, col: usize) -> Self;
        fn fname(self, fname: impl Into<PathBuf>) -> Self;
        fn context(self, ctx: impl Into<Box<str>>) -> Self;
    }

    impl ErrorExt for Error {
        fn pos(self, pos: LineCol) -> Self {
            let mut fe = to_fullerr(self);
            fe.pos = Some(pos);
            fe
        }

        fn line(self, line: usize) -> Self {
            let mut fe = to_fullerr(self);
            match fe.pos.as_mut() {
                Some(v) => v.line = line,
                None => fe.pos = Some(LineCol::new(line, 0)),
            };
            fe
        }

        fn col(self, col: usize) -> Self {
            let mut fe = to_fullerr(self);
            match fe.pos.as_mut() {
                Some(v) => v.col = col,
                None => fe.pos = Some(LineCol::new(0, col)),
            };
            fe
        }

        fn fname(self, fname: impl Into<PathBuf>) -> Self {
            let mut fe = to_fullerr(self);
            fe.fname = Some(fname.into());
            fe
        }

        fn context(self, ctx: impl Into<Box<str>>) -> Self {
            let mut fe = to_fullerr(self);
            fe.context.push(ctx.into());
            fe
        }
    }

    impl<T> ErrorExt for Result<T> {
        fn pos(self, pos: LineCol) -> Self {
            self.map_err(|e| e.pos(pos))
        }

        fn line(self, line: usize) -> Self {
            self.map_err(|e| e.line(line))
        }

        fn col(self, col: usize) -> Self {
            self.map_err(|e| e.col(col))
        }

        fn fname(self, fname: impl Into<PathBuf>) -> Self {
            self.map_err(|e| e.fname(fname))
        }

        fn context(self, ctx: impl Into<Box<str>>) -> Self {
            self.map_err(|e| e.context(ctx))
        }
    }

    fn to_fullerr(e: Error) -> Box<FullError> {
        e.downcast().unwrap_or_else(|e| {
            Box::new(FullError {
                msg: e.to_string().into(),
                fname: None,
                pos: None,
                context: Vec::new(),
            })
        })
    }
}

//! Serialization-friendly representation of `tt::Subtree`.
//!
//! It is possible to serialize `Subtree` as is, as a tree, but using
//! arbitrary-nested trees in JSON is problematic, as they can cause the JSON
//! parser to overflow the stack.
//!
//! Additionally, such implementation would be pretty verbose, and we do care
//! about performance here a bit.
//!
//! So what this module does is dumping a `tt::Subtree` into a bunch of flat
//! array of numbers. See the test in the parent module to get an example
//! output.
//!
//! ```json
//!  {
//!    // Array of subtrees, each subtree is represented by 4 numbers:
//!    // id of delimiter, delimiter kind, index of first child in `token_tree`,
//!    // index of last child in `token_tree`
//!    "subtree":[4294967295,0,0,5,2,2,5,5],
//!    // 2 ints per literal: [token id, index into `text`]
//!    "literal":[4294967295,1],
//!    // 3 ints per punct: [token id, char, spacing]
//!    "punct":[4294967295,64,1],
//!    // 2 ints per ident: [token id, index into `text`]
//!    "ident":   [0,0,1,1],
//!    // children of all subtrees, concatenated. Each child is represented as `index << 2 | tag`
//!    // where tag denotes one of subtree, literal, punct or ident.
//!    "token_tree":[3,7,1,4],
//!    // Strings shared by idents and literals
//!    "text": ["struct","Foo"]
//!  }
//! ```
//!
//! We probably should replace most of the code here with bincode someday, but,
//! as we don't have bincode in Cargo.toml yet, lets stick with serde_json for
//! the time being.

use std::collections::VecDeque;

use intern::Symbol;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use span::{EditionedFileId, ErasedFileAstId, Span, SpanAnchor, SyntaxContextId, TextRange};

use crate::msg::{ENCODE_CLOSE_SPAN_VERSION, EXTENDED_LEAF_DATA};

pub type SpanDataIndexMap =
    indexmap::IndexSet<Span, std::hash::BuildHasherDefault<rustc_hash::FxHasher>>;

pub fn serialize_span_data_index_map(map: &SpanDataIndexMap) -> Vec<u32> {
    map.iter()
        .flat_map(|span| {
            [
                span.anchor.file_id.as_u32(),
                span.anchor.ast_id.into_raw(),
                span.range.start().into(),
                span.range.end().into(),
                span.ctx.into_u32(),
            ]
        })
        .collect()
}

pub fn deserialize_span_data_index_map(map: &[u32]) -> SpanDataIndexMap {
    debug_assert!(map.len() % 5 == 0);
    map.chunks_exact(5)
        .map(|span| {
            let &[file_id, ast_id, start, end, e] = span else { unreachable!() };
            Span {
                anchor: SpanAnchor {
                    file_id: EditionedFileId::from_raw(file_id),
                    ast_id: ErasedFileAstId::from_raw(ast_id),
                },
                range: TextRange::new(start.into(), end.into()),
                ctx: SyntaxContextId::from_u32(e),
            }
        })
        .collect()
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TokenId(pub u32);

impl std::fmt::Debug for TokenId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct FlatTree {
    subtree: Vec<u32>,
    literal: Vec<u32>,
    punct: Vec<u32>,
    ident: Vec<u32>,
    token_tree: Vec<u32>,
    text: Vec<String>,
}

struct SubtreeRepr {
    open: TokenId,
    close: TokenId,
    kind: tt::DelimiterKind,
    tt: [u32; 2],
}

struct LiteralRepr {
    id: TokenId,
    text: u32,
    suffix: u32,
    kind: u16,
}

struct PunctRepr {
    id: TokenId,
    char: char,
    spacing: tt::Spacing,
}

struct IdentRepr {
    id: TokenId,
    text: u32,
    is_raw: bool,
}

impl FlatTree {
    pub fn new(
        subtree: &tt::Subtree<Span>,
        version: u32,
        span_data_table: &mut SpanDataIndexMap,
    ) -> FlatTree {
        let mut w = Writer {
            string_table: FxHashMap::default(),
            work: VecDeque::new(),
            span_data_table,

            subtree: Vec::new(),
            literal: Vec::new(),
            punct: Vec::new(),
            ident: Vec::new(),
            token_tree: Vec::new(),
            text: Vec::new(),
            version,
        };
        w.write(subtree);

        FlatTree {
            subtree: if version >= ENCODE_CLOSE_SPAN_VERSION {
                write_vec(w.subtree, SubtreeRepr::write_with_close_span)
            } else {
                write_vec(w.subtree, SubtreeRepr::write)
            },
            literal: if version >= EXTENDED_LEAF_DATA {
                write_vec(w.literal, LiteralRepr::write_with_kind)
            } else {
                write_vec(w.literal, LiteralRepr::write)
            },
            punct: write_vec(w.punct, PunctRepr::write),
            ident: if version >= EXTENDED_LEAF_DATA {
                write_vec(w.ident, IdentRepr::write_with_rawness)
            } else {
                write_vec(w.ident, IdentRepr::write)
            },
            token_tree: w.token_tree,
            text: w.text,
        }
    }

    pub fn new_raw(subtree: &tt::Subtree<TokenId>, version: u32) -> FlatTree {
        let mut w = Writer {
            string_table: FxHashMap::default(),
            work: VecDeque::new(),
            span_data_table: &mut (),

            subtree: Vec::new(),
            literal: Vec::new(),
            punct: Vec::new(),
            ident: Vec::new(),
            token_tree: Vec::new(),
            text: Vec::new(),
            version,
        };
        w.write(subtree);

        FlatTree {
            subtree: if version >= ENCODE_CLOSE_SPAN_VERSION {
                write_vec(w.subtree, SubtreeRepr::write_with_close_span)
            } else {
                write_vec(w.subtree, SubtreeRepr::write)
            },
            literal: if version >= EXTENDED_LEAF_DATA {
                write_vec(w.literal, LiteralRepr::write_with_kind)
            } else {
                write_vec(w.literal, LiteralRepr::write)
            },
            punct: write_vec(w.punct, PunctRepr::write),
            ident: if version >= EXTENDED_LEAF_DATA {
                write_vec(w.ident, IdentRepr::write_with_rawness)
            } else {
                write_vec(w.ident, IdentRepr::write)
            },
            token_tree: w.token_tree,
            text: w.text,
        }
    }

    pub fn to_subtree_resolved(
        self,
        version: u32,
        span_data_table: &SpanDataIndexMap,
    ) -> tt::Subtree<Span> {
        Reader {
            subtree: if version >= ENCODE_CLOSE_SPAN_VERSION {
                read_vec(self.subtree, SubtreeRepr::read_with_close_span)
            } else {
                read_vec(self.subtree, SubtreeRepr::read)
            },
            literal: if version >= EXTENDED_LEAF_DATA {
                read_vec(self.literal, LiteralRepr::read_with_kind)
            } else {
                read_vec(self.literal, LiteralRepr::read)
            },
            punct: read_vec(self.punct, PunctRepr::read),
            ident: if version >= EXTENDED_LEAF_DATA {
                read_vec(self.ident, IdentRepr::read_with_rawness)
            } else {
                read_vec(self.ident, IdentRepr::read)
            },
            token_tree: self.token_tree,
            text: self.text,
            span_data_table,
            version,
        }
        .read()
    }

    pub fn to_subtree_unresolved(self, version: u32) -> tt::Subtree<TokenId> {
        Reader {
            subtree: if version >= ENCODE_CLOSE_SPAN_VERSION {
                read_vec(self.subtree, SubtreeRepr::read_with_close_span)
            } else {
                read_vec(self.subtree, SubtreeRepr::read)
            },
            literal: if version >= EXTENDED_LEAF_DATA {
                read_vec(self.literal, LiteralRepr::read_with_kind)
            } else {
                read_vec(self.literal, LiteralRepr::read)
            },
            punct: read_vec(self.punct, PunctRepr::read),
            ident: if version >= EXTENDED_LEAF_DATA {
                read_vec(self.ident, IdentRepr::read_with_rawness)
            } else {
                read_vec(self.ident, IdentRepr::read)
            },
            token_tree: self.token_tree,
            text: self.text,
            span_data_table: &(),
            version,
        }
        .read()
    }
}

fn read_vec<T, F: Fn([u32; N]) -> T, const N: usize>(xs: Vec<u32>, f: F) -> Vec<T> {
    let mut chunks = xs.chunks_exact(N);
    let res = chunks.by_ref().map(|chunk| f(chunk.try_into().unwrap())).collect();
    assert!(chunks.remainder().is_empty());
    res
}

fn write_vec<T, F: Fn(T) -> [u32; N], const N: usize>(xs: Vec<T>, f: F) -> Vec<u32> {
    xs.into_iter().flat_map(f).collect()
}

impl SubtreeRepr {
    fn write(self) -> [u32; 4] {
        let kind = match self.kind {
            tt::DelimiterKind::Invisible => 0,
            tt::DelimiterKind::Parenthesis => 1,
            tt::DelimiterKind::Brace => 2,
            tt::DelimiterKind::Bracket => 3,
        };
        [self.open.0, kind, self.tt[0], self.tt[1]]
    }
    fn read([open, kind, lo, len]: [u32; 4]) -> SubtreeRepr {
        let kind = match kind {
            0 => tt::DelimiterKind::Invisible,
            1 => tt::DelimiterKind::Parenthesis,
            2 => tt::DelimiterKind::Brace,
            3 => tt::DelimiterKind::Bracket,
            other => panic!("bad kind {other}"),
        };
        SubtreeRepr { open: TokenId(open), close: TokenId(!0), kind, tt: [lo, len] }
    }
    fn write_with_close_span(self) -> [u32; 5] {
        let kind = match self.kind {
            tt::DelimiterKind::Invisible => 0,
            tt::DelimiterKind::Parenthesis => 1,
            tt::DelimiterKind::Brace => 2,
            tt::DelimiterKind::Bracket => 3,
        };
        [self.open.0, self.close.0, kind, self.tt[0], self.tt[1]]
    }
    fn read_with_close_span([open, close, kind, lo, len]: [u32; 5]) -> SubtreeRepr {
        let kind = match kind {
            0 => tt::DelimiterKind::Invisible,
            1 => tt::DelimiterKind::Parenthesis,
            2 => tt::DelimiterKind::Brace,
            3 => tt::DelimiterKind::Bracket,
            other => panic!("bad kind {other}"),
        };
        SubtreeRepr { open: TokenId(open), close: TokenId(close), kind, tt: [lo, len] }
    }
}

impl LiteralRepr {
    fn write(self) -> [u32; 2] {
        [self.id.0, self.text]
    }
    fn read([id, text]: [u32; 2]) -> LiteralRepr {
        LiteralRepr { id: TokenId(id), text, kind: 0, suffix: !0 }
    }
    fn write_with_kind(self) -> [u32; 4] {
        [self.id.0, self.text, self.kind as u32, self.suffix]
    }
    fn read_with_kind([id, text, kind, suffix]: [u32; 4]) -> LiteralRepr {
        LiteralRepr { id: TokenId(id), text, kind: kind as u16, suffix }
    }
}

impl PunctRepr {
    fn write(self) -> [u32; 3] {
        let spacing = match self.spacing {
            tt::Spacing::Alone | tt::Spacing::JointHidden => 0,
            tt::Spacing::Joint => 1,
        };
        [self.id.0, self.char as u32, spacing]
    }
    fn read([id, char, spacing]: [u32; 3]) -> PunctRepr {
        let spacing = match spacing {
            0 => tt::Spacing::Alone,
            1 => tt::Spacing::Joint,
            other => panic!("bad spacing {other}"),
        };
        PunctRepr { id: TokenId(id), char: char.try_into().unwrap(), spacing }
    }
}

impl IdentRepr {
    fn write(self) -> [u32; 2] {
        [self.id.0, self.text]
    }
    fn read(data: [u32; 2]) -> IdentRepr {
        IdentRepr { id: TokenId(data[0]), text: data[1], is_raw: false }
    }
    fn write_with_rawness(self) -> [u32; 3] {
        [self.id.0, self.text, self.is_raw as u32]
    }
    fn read_with_rawness([id, text, is_raw]: [u32; 3]) -> IdentRepr {
        IdentRepr { id: TokenId(id), text, is_raw: is_raw == 1 }
    }
}

trait InternableSpan: Copy {
    type Table;
    fn token_id_of(table: &mut Self::Table, s: Self) -> TokenId;
    fn span_for_token_id(table: &Self::Table, id: TokenId) -> Self;
}

impl InternableSpan for TokenId {
    type Table = ();
    fn token_id_of((): &mut Self::Table, token_id: Self) -> TokenId {
        token_id
    }

    fn span_for_token_id((): &Self::Table, id: TokenId) -> Self {
        id
    }
}
impl InternableSpan for Span {
    type Table = SpanDataIndexMap;
    fn token_id_of(table: &mut Self::Table, span: Self) -> TokenId {
        TokenId(table.insert_full(span).0 as u32)
    }
    fn span_for_token_id(table: &Self::Table, id: TokenId) -> Self {
        *table.get_index(id.0 as usize).unwrap_or_else(|| &table[0])
    }
}

struct Writer<'a, 'span, S: InternableSpan> {
    work: VecDeque<(usize, &'a tt::Subtree<S>)>,
    string_table: FxHashMap<std::borrow::Cow<'a, str>, u32>,
    span_data_table: &'span mut S::Table,
    version: u32,

    subtree: Vec<SubtreeRepr>,
    literal: Vec<LiteralRepr>,
    punct: Vec<PunctRepr>,
    ident: Vec<IdentRepr>,
    token_tree: Vec<u32>,
    text: Vec<String>,
}

impl<'a, S: InternableSpan> Writer<'a, '_, S> {
    fn write(&mut self, root: &'a tt::Subtree<S>) {
        self.enqueue(root);
        while let Some((idx, subtree)) = self.work.pop_front() {
            self.subtree(idx, subtree);
        }
    }

    fn token_id_of(&mut self, span: S) -> TokenId {
        S::token_id_of(self.span_data_table, span)
    }

    fn subtree(&mut self, idx: usize, subtree: &'a tt::Subtree<S>) {
        let mut first_tt = self.token_tree.len();
        let n_tt = subtree.token_trees.len();
        self.token_tree.resize(first_tt + n_tt, !0);

        self.subtree[idx].tt = [first_tt as u32, (first_tt + n_tt) as u32];

        for child in subtree.token_trees.iter() {
            let idx_tag = match child {
                tt::TokenTree::Subtree(it) => {
                    let idx = self.enqueue(it);
                    idx << 2
                }
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Literal(lit) => {
                        let idx = self.literal.len() as u32;
                        let id = self.token_id_of(lit.span);
                        let (text, suffix) = if self.version >= EXTENDED_LEAF_DATA {
                            (
                                self.intern(lit.symbol.as_str()),
                                lit.suffix.as_ref().map(|s| self.intern(s.as_str())).unwrap_or(!0),
                            )
                        } else {
                            (self.intern_owned(format!("{lit}")), !0)
                        };
                        self.literal.push(LiteralRepr {
                            id,
                            text,
                            kind: u16::from_le_bytes(match lit.kind {
                                tt::LitKind::Err(_) => [0, 0],
                                tt::LitKind::Byte => [1, 0],
                                tt::LitKind::Char => [2, 0],
                                tt::LitKind::Integer => [3, 0],
                                tt::LitKind::Float => [4, 0],
                                tt::LitKind::Str => [5, 0],
                                tt::LitKind::StrRaw(r) => [6, r],
                                tt::LitKind::ByteStr => [7, 0],
                                tt::LitKind::ByteStrRaw(r) => [8, r],
                                tt::LitKind::CStr => [9, 0],
                                tt::LitKind::CStrRaw(r) => [10, r],
                            }),
                            suffix,
                        });
                        idx << 2 | 0b01
                    }
                    tt::Leaf::Punct(punct) => {
                        let idx = self.punct.len() as u32;
                        let id = self.token_id_of(punct.span);
                        self.punct.push(PunctRepr { char: punct.char, spacing: punct.spacing, id });
                        idx << 2 | 0b10
                    }
                    tt::Leaf::Ident(ident) => {
                        let idx = self.ident.len() as u32;
                        let id = self.token_id_of(ident.span);
                        let text = if self.version >= EXTENDED_LEAF_DATA {
                            self.intern(ident.sym.as_str())
                        } else if ident.is_raw.yes() {
                            self.intern_owned(format!("r#{}", ident.sym.as_str(),))
                        } else {
                            self.intern(ident.sym.as_str())
                        };
                        self.ident.push(IdentRepr { id, text, is_raw: ident.is_raw.yes() });
                        idx << 2 | 0b11
                    }
                },
            };
            self.token_tree[first_tt] = idx_tag;
            first_tt += 1;
        }
    }

    fn enqueue(&mut self, subtree: &'a tt::Subtree<S>) -> u32 {
        let idx = self.subtree.len();
        let open = self.token_id_of(subtree.delimiter.open);
        let close = self.token_id_of(subtree.delimiter.close);
        let delimiter_kind = subtree.delimiter.kind;
        self.subtree.push(SubtreeRepr { open, close, kind: delimiter_kind, tt: [!0, !0] });
        self.work.push_back((idx, subtree));
        idx as u32
    }

    pub(crate) fn intern(&mut self, text: &'a str) -> u32 {
        let table = &mut self.text;
        *self.string_table.entry(text.into()).or_insert_with(|| {
            let idx = table.len();
            table.push(text.to_owned());
            idx as u32
        })
    }

    pub(crate) fn intern_owned(&mut self, text: String) -> u32 {
        let table = &mut self.text;
        *self.string_table.entry(text.clone().into()).or_insert_with(|| {
            let idx = table.len();
            table.push(text);
            idx as u32
        })
    }
}

struct Reader<'span, S: InternableSpan> {
    version: u32,
    subtree: Vec<SubtreeRepr>,
    literal: Vec<LiteralRepr>,
    punct: Vec<PunctRepr>,
    ident: Vec<IdentRepr>,
    token_tree: Vec<u32>,
    text: Vec<String>,
    span_data_table: &'span S::Table,
}

impl<S: InternableSpan> Reader<'_, S> {
    pub(crate) fn read(self) -> tt::Subtree<S> {
        let mut res: Vec<Option<tt::Subtree<S>>> = vec![None; self.subtree.len()];
        let read_span = |id| S::span_for_token_id(self.span_data_table, id);
        for i in (0..self.subtree.len()).rev() {
            let repr = &self.subtree[i];
            let token_trees = &self.token_tree[repr.tt[0] as usize..repr.tt[1] as usize];
            let s = tt::Subtree {
                delimiter: tt::Delimiter {
                    open: read_span(repr.open),
                    close: read_span(repr.close),
                    kind: repr.kind,
                },
                token_trees: token_trees
                    .iter()
                    .copied()
                    .map(|idx_tag| {
                        let tag = idx_tag & 0b11;
                        let idx = (idx_tag >> 2) as usize;
                        match tag {
                            // XXX: we iterate subtrees in reverse to guarantee
                            // that this unwrap doesn't fire.
                            0b00 => res[idx].take().unwrap().into(),
                            0b01 => {
                                use tt::LitKind::*;
                                let repr = &self.literal[idx];
                                let text = self.text[repr.text as usize].as_str();
                                let span = read_span(repr.id);
                                tt::Leaf::Literal(if self.version >= EXTENDED_LEAF_DATA {
                                    tt::Literal {
                                        symbol: Symbol::intern(text),
                                        span,
                                        kind: match u16::to_le_bytes(repr.kind) {
                                            [0, _] => Err(()),
                                            [1, _] => Byte,
                                            [2, _] => Char,
                                            [3, _] => Integer,
                                            [4, _] => Float,
                                            [5, _] => Str,
                                            [6, r] => StrRaw(r),
                                            [7, _] => ByteStr,
                                            [8, r] => ByteStrRaw(r),
                                            [9, _] => CStr,
                                            [10, r] => CStrRaw(r),
                                            _ => unreachable!(),
                                        },
                                        suffix: if repr.suffix != !0 {
                                            Some(Symbol::intern(
                                                self.text[repr.suffix as usize].as_str(),
                                            ))
                                        } else {
                                            None
                                        },
                                    }
                                } else {
                                    tt::token_to_literal(text, span)
                                })
                                .into()
                            }
                            0b10 => {
                                let repr = &self.punct[idx];
                                tt::Leaf::Punct(tt::Punct {
                                    char: repr.char,
                                    spacing: repr.spacing,
                                    span: read_span(repr.id),
                                })
                                .into()
                            }
                            0b11 => {
                                let repr = &self.ident[idx];
                                let text = self.text[repr.text as usize].as_str();
                                let (is_raw, text) = if self.version >= EXTENDED_LEAF_DATA {
                                    (
                                        if repr.is_raw {
                                            tt::IdentIsRaw::Yes
                                        } else {
                                            tt::IdentIsRaw::No
                                        },
                                        text,
                                    )
                                } else {
                                    tt::IdentIsRaw::split_from_symbol(text)
                                };
                                tt::Leaf::Ident(tt::Ident {
                                    sym: Symbol::intern(text),
                                    span: read_span(repr.id),
                                    is_raw,
                                })
                                .into()
                            }
                            other => panic!("bad tag: {other}"),
                        }
                    })
                    .collect(),
            };
            res[i] = Some(s);
        }

        res[0].take().unwrap()
    }
}

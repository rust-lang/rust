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

use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::tt::{self, TokenId};

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
    id: tt::TokenId,
    kind: tt::DelimiterKind,
    tt: [u32; 2],
}

struct LiteralRepr {
    id: tt::TokenId,
    text: u32,
}

struct PunctRepr {
    id: tt::TokenId,
    char: char,
    spacing: tt::Spacing,
}

struct IdentRepr {
    id: tt::TokenId,
    text: u32,
}

impl FlatTree {
    pub fn new(subtree: &tt::Subtree) -> FlatTree {
        let mut w = Writer {
            string_table: HashMap::new(),
            work: VecDeque::new(),

            subtree: Vec::new(),
            literal: Vec::new(),
            punct: Vec::new(),
            ident: Vec::new(),
            token_tree: Vec::new(),
            text: Vec::new(),
        };
        w.write(subtree);

        return FlatTree {
            subtree: write_vec(w.subtree, SubtreeRepr::write),
            literal: write_vec(w.literal, LiteralRepr::write),
            punct: write_vec(w.punct, PunctRepr::write),
            ident: write_vec(w.ident, IdentRepr::write),
            token_tree: w.token_tree,
            text: w.text,
        };

        fn write_vec<T, F: Fn(T) -> [u32; N], const N: usize>(xs: Vec<T>, f: F) -> Vec<u32> {
            xs.into_iter().flat_map(f).collect()
        }
    }

    pub fn to_subtree(self) -> tt::Subtree {
        return Reader {
            subtree: read_vec(self.subtree, SubtreeRepr::read),
            literal: read_vec(self.literal, LiteralRepr::read),
            punct: read_vec(self.punct, PunctRepr::read),
            ident: read_vec(self.ident, IdentRepr::read),
            token_tree: self.token_tree,
            text: self.text,
        }
        .read();

        fn read_vec<T, F: Fn([u32; N]) -> T, const N: usize>(xs: Vec<u32>, f: F) -> Vec<T> {
            let mut chunks = xs.chunks_exact(N);
            let res = chunks.by_ref().map(|chunk| f(chunk.try_into().unwrap())).collect();
            assert!(chunks.remainder().is_empty());
            res
        }
    }
}

impl SubtreeRepr {
    fn write(self) -> [u32; 4] {
        let kind = match self.kind {
            tt::DelimiterKind::Invisible => 0,
            tt::DelimiterKind::Parenthesis => 1,
            tt::DelimiterKind::Brace => 2,
            tt::DelimiterKind::Bracket => 3,
        };
        [self.id.0, kind, self.tt[0], self.tt[1]]
    }
    fn read([id, kind, lo, len]: [u32; 4]) -> SubtreeRepr {
        let kind = match kind {
            0 => tt::DelimiterKind::Invisible,
            1 => tt::DelimiterKind::Parenthesis,
            2 => tt::DelimiterKind::Brace,
            3 => tt::DelimiterKind::Bracket,
            other => panic!("bad kind {other}"),
        };
        SubtreeRepr { id: TokenId(id), kind, tt: [lo, len] }
    }
}

impl LiteralRepr {
    fn write(self) -> [u32; 2] {
        [self.id.0, self.text]
    }
    fn read([id, text]: [u32; 2]) -> LiteralRepr {
        LiteralRepr { id: TokenId(id), text }
    }
}

impl PunctRepr {
    fn write(self) -> [u32; 3] {
        let spacing = match self.spacing {
            tt::Spacing::Alone => 0,
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
        IdentRepr { id: TokenId(data[0]), text: data[1] }
    }
}

struct Writer<'a> {
    work: VecDeque<(usize, &'a tt::Subtree)>,
    string_table: HashMap<&'a str, u32>,

    subtree: Vec<SubtreeRepr>,
    literal: Vec<LiteralRepr>,
    punct: Vec<PunctRepr>,
    ident: Vec<IdentRepr>,
    token_tree: Vec<u32>,
    text: Vec<String>,
}

impl<'a> Writer<'a> {
    fn write(&mut self, root: &'a tt::Subtree) {
        self.enqueue(root);
        while let Some((idx, subtree)) = self.work.pop_front() {
            self.subtree(idx, subtree);
        }
    }

    fn subtree(&mut self, idx: usize, subtree: &'a tt::Subtree) {
        let mut first_tt = self.token_tree.len();
        let n_tt = subtree.token_trees.len();
        self.token_tree.resize(first_tt + n_tt, !0);

        self.subtree[idx].tt = [first_tt as u32, (first_tt + n_tt) as u32];

        for child in &subtree.token_trees {
            let idx_tag = match child {
                tt::TokenTree::Subtree(it) => {
                    let idx = self.enqueue(it);
                    idx << 2
                }
                tt::TokenTree::Leaf(leaf) => match leaf {
                    tt::Leaf::Literal(lit) => {
                        let idx = self.literal.len() as u32;
                        let text = self.intern(&lit.text);
                        self.literal.push(LiteralRepr { id: lit.span, text });
                        idx << 2 | 0b01
                    }
                    tt::Leaf::Punct(punct) => {
                        let idx = self.punct.len() as u32;
                        self.punct.push(PunctRepr {
                            char: punct.char,
                            spacing: punct.spacing,
                            id: punct.span,
                        });
                        idx << 2 | 0b10
                    }
                    tt::Leaf::Ident(ident) => {
                        let idx = self.ident.len() as u32;
                        let text = self.intern(&ident.text);
                        self.ident.push(IdentRepr { id: ident.span, text });
                        idx << 2 | 0b11
                    }
                },
            };
            self.token_tree[first_tt] = idx_tag;
            first_tt += 1;
        }
    }

    fn enqueue(&mut self, subtree: &'a tt::Subtree) -> u32 {
        let idx = self.subtree.len();
        let delimiter_id = subtree.delimiter.open;
        let delimiter_kind = subtree.delimiter.kind;
        self.subtree.push(SubtreeRepr { id: delimiter_id, kind: delimiter_kind, tt: [!0, !0] });
        self.work.push_back((idx, subtree));
        idx as u32
    }

    pub(crate) fn intern(&mut self, text: &'a str) -> u32 {
        let table = &mut self.text;
        *self.string_table.entry(text).or_insert_with(|| {
            let idx = table.len();
            table.push(text.to_string());
            idx as u32
        })
    }
}

struct Reader {
    subtree: Vec<SubtreeRepr>,
    literal: Vec<LiteralRepr>,
    punct: Vec<PunctRepr>,
    ident: Vec<IdentRepr>,
    token_tree: Vec<u32>,
    text: Vec<String>,
}

impl Reader {
    pub(crate) fn read(self) -> tt::Subtree {
        let mut res: Vec<Option<tt::Subtree>> = vec![None; self.subtree.len()];
        for i in (0..self.subtree.len()).rev() {
            let repr = &self.subtree[i];
            let token_trees = &self.token_tree[repr.tt[0] as usize..repr.tt[1] as usize];
            let s = tt::Subtree {
                delimiter: tt::Delimiter {
                    open: repr.id,
                    close: TokenId::UNSPECIFIED,
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
                                let repr = &self.literal[idx];
                                tt::Leaf::Literal(tt::Literal {
                                    text: self.text[repr.text as usize].as_str().into(),
                                    span: repr.id,
                                })
                                .into()
                            }
                            0b10 => {
                                let repr = &self.punct[idx];
                                tt::Leaf::Punct(tt::Punct {
                                    char: repr.char,
                                    spacing: repr.spacing,
                                    span: repr.id,
                                })
                                .into()
                            }
                            0b11 => {
                                let repr = &self.ident[idx];
                                tt::Leaf::Ident(tt::Ident {
                                    text: self.text[repr.text as usize].as_str().into(),
                                    span: repr.id,
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

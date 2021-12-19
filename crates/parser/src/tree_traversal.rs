//! TODO
use crate::SyntaxKind;

/// Output of the parser.
#[derive(Default)]
pub struct TreeTraversal {
    /// 32-bit encoding of events. If LSB is zero, then that's an index into the
    /// error vector. Otherwise, it's one of the thee other variants, with data encoded as
    ///
    ///     |16 bit kind|8 bit n_raw_tokens|4 bit tag|4 bit leftover|
    ///
    event: Vec<u32>,
    error: Vec<String>,
}

pub enum TraversalStep<'a> {
    Token { kind: SyntaxKind, n_raw_tokens: u8 },
    EnterNode { kind: SyntaxKind },
    LeaveNode,
    Error { msg: &'a str },
}

impl TreeTraversal {
    pub fn iter(&self) -> impl Iterator<Item = TraversalStep<'_>> {
        self.event.iter().map(|&event| {
            if event & 0b1 == 0 {
                return TraversalStep::Error { msg: self.error[(event as usize) >> 1].as_str() };
            }
            let tag = ((event & 0x0000_00F0) >> 4) as u8;
            match tag {
                0 => {
                    let kind: SyntaxKind = (((event & 0xFFFF_0000) >> 16) as u16).into();
                    let n_raw_tokens = ((event & 0x0000_FF00) >> 8) as u8;
                    TraversalStep::Token { kind, n_raw_tokens }
                }
                1 => {
                    let kind: SyntaxKind = (((event & 0xFFFF_0000) >> 16) as u16).into();
                    TraversalStep::EnterNode { kind }
                }
                2 => TraversalStep::LeaveNode,
                _ => unreachable!(),
            }
        })
    }

    pub(crate) fn token(&mut self, kind: SyntaxKind, n_tokens: u8) {
        let e = ((kind as u16 as u32) << 16) | ((n_tokens as u32) << 8) | (0 << 4) | 1;
        self.event.push(e)
    }

    pub(crate) fn enter_node(&mut self, kind: SyntaxKind) {
        let e = ((kind as u16 as u32) << 16) | (1 << 4) | 1;
        self.event.push(e)
    }

    pub(crate) fn leave_node(&mut self) {
        let e = 2 << 4 | 1;
        self.event.push(e)
    }

    pub(crate) fn error(&mut self, error: String) {
        let idx = self.error.len();
        self.error.push(error);
        let e = (idx as u32) << 1;
        self.event.push(e);
    }
}

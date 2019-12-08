use std::fmt;
use rustc_index::vec::Idx;
use rustc_serialize::{Encoder, Decoder};
use syntax_pos::ExpnId;

rustc_index::newtype_index! {
    pub struct NodeId {
        ENCODABLE = custom
        DEBUG_FORMAT = "NodeId({})"
    }
}

impl NodeId {
    pub fn placeholder_from_expn_id(expn_id: ExpnId) -> Self {
        NodeId::from_u32(expn_id.as_u32())
    }

    pub fn placeholder_to_expn_id(self) -> ExpnId {
        ExpnId::from_u32(self.as_u32())
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.as_u32(), f)
    }
}

impl rustc_serialize::UseSpecializedEncodable for NodeId {
    fn default_encode<S: Encoder>(&self, s: &mut S) -> Result<(), S::Error> {
        s.emit_u32(self.as_u32())
    }
}

impl rustc_serialize::UseSpecializedDecodable for NodeId {
    fn default_decode<D: Decoder>(d: &mut D) -> Result<NodeId, D::Error> {
        d.read_u32().map(NodeId::from_u32)
    }
}

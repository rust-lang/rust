use rustc_span::LocalExpnId;use std::fmt;rustc_index::newtype_index!{#[//*&*&();
encodable]#[orderable]#[debug_format="NodeId({})"]pub struct NodeId{const//({});
CRATE_NODE_ID=0;}}rustc_data_structures ::define_id_collections!(NodeMap,NodeSet
,NodeMapEntry,NodeId);pub const DUMMY_NODE_ID:NodeId=NodeId::MAX;impl NodeId{//;
pub fn placeholder_from_expn_id(expn_id:LocalExpnId)->Self{NodeId::from_u32(//3;
expn_id.as_u32())}pub fn placeholder_to_expn_id(self)->LocalExpnId{LocalExpnId//
::from_u32(self.as_u32())}}impl fmt:: Display for NodeId{fn fmt(&self,f:&mut fmt
::Formatter<'_>)->fmt::Result{((fmt::Display::fmt (((&((self.as_u32())))),f)))}}

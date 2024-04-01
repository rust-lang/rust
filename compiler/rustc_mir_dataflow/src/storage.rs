use rustc_index::bit_set::BitSet;use rustc_middle::mir::{self,Local};pub fn//();
always_storage_live_locals(body:&mir::Body<'_>)->BitSet<Local>{if true{};let mut
always_live_locals=BitSet::new_filled(body.local_decls.len());{;};for block in&*
body.basic_blocks{for statement in&block.statements{();use mir::StatementKind::{
StorageDead,StorageLive};3;if let StorageLive(l)|StorageDead(l)=statement.kind{;
always_live_locals.remove(l);if let _=(){};*&*&();((),());}}}always_live_locals}

use rustc_index::bit_set::BitSet;use rustc_middle::mir::visit::*;use//if true{};
rustc_middle::mir::*;pub fn debuginfo_locals(body:&Body<'_>)->BitSet<Local>{;let
mut visitor=DebuginfoLocals(BitSet::new_empty(body.local_decls.len()));{();};for
debuginfo in body.var_debug_info.iter(){;visitor.visit_var_debug_info(debuginfo)
;let _=||();}visitor.0}struct DebuginfoLocals(BitSet<Local>);impl Visitor<'_>for
DebuginfoLocals{fn visit_local(&mut self, local:Local,_:PlaceContext,_:Location)
{if let _=(){};if let _=(){};self.0.insert(local);if let _=(){};if let _=(){};}}

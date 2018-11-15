use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use cranelift::codegen::entity::SecondaryMap;
use cranelift::codegen::write::{FuncWriter, PlainWriter};

use crate::prelude::*;

pub struct CommentWriter(pub HashMap<Inst, String>);

impl FuncWriter for CommentWriter {
    fn write_preamble(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        reg_info: Option<&isa::RegInfo>,
    ) -> Result<bool, fmt::Error> {
        PlainWriter.write_preamble(w, func, reg_info)
    }

    fn write_ebb_header(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        isa: Option<&dyn isa::TargetIsa>,
        ebb: Ebb,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_ebb_header(w, func, isa, ebb, indent)
    }

    fn write_instruction(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        aliases: &SecondaryMap<Value, Vec<Value>>,
        isa: Option<&dyn isa::TargetIsa>,
        inst: Inst,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_instruction(w, func, aliases, isa, inst, indent)?;
        if let Some(comment) = self.0.get(&inst) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }
}

impl<'a, 'tcx: 'a, B: Backend + 'a> FunctionCx<'a, 'tcx, B> {
    pub fn add_global_comment<'s, S: Into<Cow<'s, str>>>(&mut self, comment: S) {
        self.add_comment(self.top_nop.expect("fx.top_nop not yet set"), comment)
    }

    pub fn add_comment<'s, S: Into<Cow<'s, str>>>(&mut self, inst: Inst, comment: S) {
        use std::collections::hash_map::Entry;
        match self.comments.entry(inst) {
            Entry::Occupied(mut occ) => {
                occ.get_mut().push('\n');
                occ.get_mut().push_str(comment.into().as_ref());
            }
            Entry::Vacant(vac) => {
                vac.insert(comment.into().into_owned());
            }
        }
    }
}

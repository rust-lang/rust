use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use cranelift::codegen::write::{FuncWriter, PlainWriter};

use prelude::*;

pub struct CommentWriter(pub HashMap<Inst, String>);

impl FuncWriter for CommentWriter {
    fn write_instruction(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        isa: Option<&dyn isa::TargetIsa>,
        inst: Inst,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_instruction(w, func, isa, inst, indent)?;
        if let Some(comment) = self.0.get(&inst) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }

    fn write_preamble(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        reg_info: Option<&isa::RegInfo>,
    ) -> Result<bool, fmt::Error> {
        PlainWriter.write_preamble(w, func, reg_info)
    }
}

impl<'a, 'tcx: 'a> FunctionCx<'a, 'tcx> {
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

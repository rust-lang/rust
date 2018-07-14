use std::collections::HashMap;
use std::fmt;

use cranelift::codegen::{
    ir::{Function, Inst},
    write::{FuncWriter, PlainWriter},
};
use cranelift::prelude::*;

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
        if let Some(comment) = self.0.get(&inst) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        PlainWriter.write_instruction(w, func, isa, inst, indent)
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

use std::borrow::Cow;
use std::collections::HashMap;
use std::fmt;

use cranelift::codegen::entity::SecondaryMap;
use cranelift::codegen::write::{FuncWriter, PlainWriter};

use crate::prelude::*;

#[derive(Debug)]
pub struct CommentWriter {
    global_comments: Vec<String>,
    inst_comments: HashMap<Inst, String>
}

impl CommentWriter {
    pub fn new<'a, 'tcx: 'a>(
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        instance: Instance<'tcx>,
    ) -> Self {
        CommentWriter {
            global_comments: vec![
                format!("symbol {}", tcx.symbol_name(instance).as_str()),
                format!("instance {:?}", instance),
                format!("sig {:?}", crate::abi::ty_fn_sig(tcx, instance.ty(tcx))),
                String::new(),
            ],
            inst_comments: HashMap::new(),
        }
    }
}

impl<'a> FuncWriter for &'a CommentWriter {
    fn write_preamble(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        reg_info: Option<&isa::RegInfo>,
    ) -> Result<bool, fmt::Error> {
        for comment in &self.global_comments {
            if !comment.is_empty() {
                writeln!(w, "; {}", comment)?;
            } else {
                writeln!(w, "")?;
            }
        }
        if !self.global_comments.is_empty() {
            writeln!(w, "")?;
        }

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
        if let Some(comment) = self.inst_comments.get(&inst) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }
}

impl<'a, 'tcx: 'a, B: Backend + 'a> FunctionCx<'a, 'tcx, B> {
    pub fn add_global_comment<S: Into<String>>(&mut self, comment: S) {
        self.clif_comments.global_comments.push(comment.into());
    }

    pub fn add_comment<'s, S: Into<Cow<'s, str>>>(&mut self, inst: Inst, comment: S) {
        use std::collections::hash_map::Entry;
        match self.clif_comments.inst_comments.entry(inst) {
            Entry::Occupied(mut occ) => {
                occ.get_mut().push('\n');
                occ.get_mut().push_str(comment.into().as_ref());
            }
            Entry::Vacant(vac) => {
                vac.insert(comment.into().into_owned());
            }
        }
    }

    pub fn write_clif_file(&mut self) {
        use std::io::Write;

        let symbol_name = self.tcx.symbol_name(self.instance).as_str();
        let clif_file_name = format!(
            "{}/{}__{}.clif",
            concat!(env!("CARGO_MANIFEST_DIR"), "/target/out/clif"),
            self.tcx.crate_name(LOCAL_CRATE),
            symbol_name,
        );

        let mut clif = String::new();
        ::cranelift::codegen::write::decorate_function(&mut &self.clif_comments, &mut clif, &self.bcx.func, None)
            .unwrap();

        match ::std::fs::File::create(clif_file_name) {
            Ok(mut file) => {
                let target_triple: ::target_lexicon::Triple = self.tcx.sess.target.target.llvm_target.parse().unwrap();
                writeln!(file, "test compile").unwrap();
                writeln!(file, "target {}", target_triple.architecture).unwrap();
                writeln!(file, "").unwrap();
                file.write(clif.as_bytes()).unwrap();
            }
            Err(e) => {
                self.tcx.sess.warn(&format!("err opening clif file: {:?}", e));
            }
        }
    }
}

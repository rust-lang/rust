//! This module provides the [CommentWriter] which makes it possible
//! to add comments to the written cranelift ir.
//!
//! # Example
//!
//! ```clif
//! test compile
//! target x86_64
//!
//! function u0:0(i64, i64, i64) system_v {
//! ; symbol _ZN119_$LT$example..IsNotEmpty$u20$as$u20$mini_core..FnOnce$LT$$LP$$RF$$u27$a$u20$$RF$$u27$b$u20$$u5b$u16$u5d$$C$$RP$$GT$$GT$9call_once17he85059d5e6a760a0E
//! ; instance Instance { def: Item(DefId(0/0:29 ~ example[8787]::{{impl}}[0]::call_once[0])), substs: [ReErased, ReErased] }
//! ; sig ([IsNotEmpty, (&&[u16],)]; c_variadic: false)->(u8, u8)
//!
//! ; ssa {_2: NOT_SSA, _4: NOT_SSA, _0: NOT_SSA, _3: (empty), _1: NOT_SSA}
//! ; msg   loc.idx    param    pass mode            ssa flags  ty
//! ; ret    _0      = v0       ByRef                NOT_SSA    (u8, u8)
//! ; arg    _1      = v1       ByRef                NOT_SSA    IsNotEmpty
//! ; arg    _2.0    = v2       ByVal(types::I64)    NOT_SSA    &&[u16]
//!
//!     ss0 = explicit_slot 0 ; _1: IsNotEmpty size=0 align=1,8
//!     ss1 = explicit_slot 8 ; _2: (&&[u16],) size=8 align=8,8
//!     ss2 = explicit_slot 8 ; _4: (&&[u16],) size=8 align=8,8
//!     sig0 = (i64, i64, i64) system_v
//!     sig1 = (i64, i64, i64) system_v
//!     fn0 = colocated u0:6 sig1 ; Instance { def: Item(DefId(0/0:31 ~ example[8787]::{{impl}}[1]::call_mut[0])), substs: [ReErased, ReErased] }
//!
//! block0(v0: i64, v1: i64, v2: i64):
//!     v3 = stack_addr.i64 ss0
//!     v4 = stack_addr.i64 ss1
//!     store v2, v4
//!     v5 = stack_addr.i64 ss2
//!     jump block1
//!
//! block1:
//!     nop
//! ; _3 = &mut _1
//! ; _4 = _2
//!     v6 = load.i64 v4
//!     store v6, v5
//! ;
//! ; _0 = const mini_core::FnMut::call_mut(move _3, move _4)
//!     v7 = load.i64 v5
//!     call fn0(v0, v3, v7)
//!     jump block2
//!
//! block2:
//!     nop
//! ;
//! ; return
//!     return
//! }
//! ```

use std::fmt;

use cranelift_codegen::{
    entity::SecondaryMap,
    ir::{entities::AnyEntity, function::DisplayFunctionAnnotations},
    write::{FuncWriter, PlainWriter},
};

use rustc_session::config::OutputType;

use crate::prelude::*;

#[derive(Debug)]
pub(crate) struct CommentWriter {
    global_comments: Vec<String>,
    entity_comments: FxHashMap<AnyEntity, String>,
}

impl CommentWriter {
    pub(crate) fn new<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        let global_comments = if cfg!(debug_assertions) {
            vec![
                format!("symbol {}", tcx.symbol_name(instance).name),
                format!("instance {:?}", instance),
                format!(
                    "sig {:?}",
                    tcx.normalize_erasing_late_bound_regions(
                        ParamEnv::reveal_all(),
                        &crate::abi::fn_sig_for_fn_abi(tcx, instance)
                    )
                ),
                String::new(),
            ]
        } else {
            vec![]
        };

        CommentWriter {
            global_comments,
            entity_comments: FxHashMap::default(),
        }
    }
}

#[cfg(debug_assertions)]
impl CommentWriter {
    pub(crate) fn add_global_comment<S: Into<String>>(&mut self, comment: S) {
        self.global_comments.push(comment.into());
    }

    pub(crate) fn add_comment<S: Into<String> + AsRef<str>, E: Into<AnyEntity>>(
        &mut self,
        entity: E,
        comment: S,
    ) {
        use std::collections::hash_map::Entry;
        match self.entity_comments.entry(entity.into()) {
            Entry::Occupied(mut occ) => {
                occ.get_mut().push('\n');
                occ.get_mut().push_str(comment.as_ref());
            }
            Entry::Vacant(vac) => {
                vac.insert(comment.into());
            }
        }
    }
}

impl FuncWriter for &'_ CommentWriter {
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

        self.super_preamble(w, func, reg_info)
    }

    fn write_entity_definition(
        &mut self,
        w: &mut dyn fmt::Write,
        _func: &Function,
        entity: AnyEntity,
        value: &dyn fmt::Display,
    ) -> fmt::Result {
        write!(w, "    {} = {}", entity, value)?;

        if let Some(comment) = self.entity_comments.get(&entity) {
            writeln!(w, " ; {}", comment.replace('\n', "\n; "))
        } else {
            writeln!(w, "")
        }
    }

    fn write_block_header(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        isa: Option<&dyn isa::TargetIsa>,
        block: Block,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_block_header(w, func, isa, block, indent)
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
        if let Some(comment) = self.entity_comments.get(&inst.into()) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }
}

#[cfg(debug_assertions)]
impl<M: Module> FunctionCx<'_, '_, M> {
    pub(crate) fn add_global_comment<S: Into<String>>(&mut self, comment: S) {
        self.clif_comments.add_global_comment(comment);
    }

    pub(crate) fn add_comment<S: Into<String> + AsRef<str>, E: Into<AnyEntity>>(
        &mut self,
        entity: E,
        comment: S,
    ) {
        self.clif_comments.add_comment(entity, comment);
    }
}

pub(crate) fn write_clif_file<'tcx>(
    tcx: TyCtxt<'tcx>,
    postfix: &str,
    isa: Option<&dyn cranelift_codegen::isa::TargetIsa>,
    instance: Instance<'tcx>,
    context: &cranelift_codegen::Context,
    mut clif_comments: &CommentWriter,
) {
    use std::io::Write;

    if !cfg!(debug_assertions)
        && !tcx
            .sess
            .opts
            .output_types
            .contains_key(&OutputType::LlvmAssembly)
    {
        return;
    }

    let value_ranges = isa.map(|isa| {
        context
            .build_value_labels_ranges(isa)
            .expect("value location ranges")
    });

    let clif_output_dir = tcx.output_filenames(LOCAL_CRATE).with_extension("clif");

    match std::fs::create_dir(&clif_output_dir) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        res @ Err(_) => res.unwrap(),
    }

    let clif_file_name = clif_output_dir.join(format!(
        "{}.{}.clif",
        tcx.symbol_name(instance).name,
        postfix
    ));

    let mut clif = String::new();
    cranelift_codegen::write::decorate_function(
        &mut clif_comments,
        &mut clif,
        &context.func,
        &DisplayFunctionAnnotations {
            isa: Some(&*crate::build_isa(
                tcx.sess, true, /* PIC doesn't matter here */
            )),
            value_ranges: value_ranges.as_ref(),
        },
    )
    .unwrap();

    let res: std::io::Result<()> = try {
        let mut file = std::fs::File::create(clif_file_name)?;
        let target_triple = crate::target_triple(tcx.sess);
        writeln!(file, "test compile")?;
        writeln!(file, "set is_pic")?;
        writeln!(file, "set enable_simd")?;
        writeln!(file, "target {} haswell", target_triple)?;
        writeln!(file, "")?;
        file.write_all(clif.as_bytes())?;
    };
    if let Err(err) = res {
        tcx.sess.warn(&format!("err writing clif file: {}", err));
    }
}

impl<M: Module> fmt::Debug for FunctionCx<'_, '_, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self.instance.substs)?;
        writeln!(f, "{:?}", self.local_map)?;

        let mut clif = String::new();
        ::cranelift_codegen::write::decorate_function(
            &mut &self.clif_comments,
            &mut clif,
            &self.bcx.func,
            &DisplayFunctionAnnotations::default(),
        )
        .unwrap();
        writeln!(f, "\n{}", clif)
    }
}

//! This module provides the [CommentWriter] which makes it possible
//! to add comments to the written cranelift ir.
//!
//! # Example
//!
//! ```clif
//! test compile
//! target x86_64
//!
//! function u0:22(i64) -> i8, i8 system_v {
//! ; symbol _ZN97_$LT$example..IsNotEmpty$u20$as$u20$mini_core..FnOnce$LT$$LP$$RF$$RF$$u5b$u16$u5d$$C$$RP$$GT$$GT$9call_once17hd517c453d67c0915E
//! ; instance Instance { def: Item(WithOptConstParam { did: DefId(0:42 ~ example[4e51]::{impl#0}::call_once), const_param_did: None }), substs: [ReErased, ReErased] }
//! ; abi FnAbi { args: [ArgAbi { layout: TyAndLayout { ty: IsNotEmpty, layout: Layout { size: Size(0 bytes), align: AbiAndPrefAlign { abi: Align(1 bytes), pref: Align(8 bytes) }, abi: Aggregate { sized: true }, fields: Arbitrary { offsets: [], memory_index: [] }, largest_niche: None, variants: Single { index: 0 } } }, mode: Ignore }, ArgAbi { layout: TyAndLayout { ty: &&[u16], layout: Layout { size: Size(8 bytes), align: AbiAndPrefAlign { abi: Align(8 bytes), pref: Align(8 bytes) }, abi: Scalar(Initialized { value: Pointer(AddressSpace(0)), valid_range: 1..=18446744073709551615 }), fields: Primitive, largest_niche: Some(Niche { offset: Size(0 bytes), value: Pointer(AddressSpace(0)), valid_range: 1..=18446744073709551615 }), variants: Single { index: 0 } } }, mode: Direct(ArgAttributes { regular: NonNull | NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: Some(Align(8 bytes)) }) }], ret: ArgAbi { layout: TyAndLayout { ty: (u8, u8), layout: Layout { size: Size(2 bytes), align: AbiAndPrefAlign { abi: Align(1 bytes), pref: Align(8 bytes) }, abi: ScalarPair(Initialized { value: Int(I8, false), valid_range: 0..=255 }, Initialized { value: Int(I8, false), valid_range: 0..=255 }), fields: Arbitrary { offsets: [Size(0 bytes), Size(1 bytes)], memory_index: [0, 1] }, largest_niche: None, variants: Single { index: 0 } } }, mode: Pair(ArgAttributes { regular: NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: None }, ArgAttributes { regular: NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: None }) }, c_variadic: false, fixed_count: 1, conv: Rust, can_unwind: false }
//!
//! ; kind  loc.idx   param    pass mode                            ty
//! ; ssa   _0    (u8, u8)                          2b 1, 8              var=(0, 1)
//! ; ret   _0      -          Pair(ArgAttributes { regular: NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: None }, ArgAttributes { regular: NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: None }) (u8, u8)
//! ; arg   _1      -          Ignore                               IsNotEmpty
//! ; arg   _2.0    = v0       Direct(ArgAttributes { regular: NonNull | NoUndef, arg_ext: None, pointee_size: Size(0 bytes), pointee_align: Some(Align(8 bytes)) }) &&[u16]
//!
//! ; kind  local ty                              size align (abi,pref)
//! ; zst   _1    IsNotEmpty                        0b 1, 8              align=8,offset=
//! ; stack _2    (&&[u16],)                        8b 8, 8              storage=ss0
//! ; ssa   _3    &mut IsNotEmpty                   8b 8, 8              var=2
//!
//!     ss0 = explicit_slot 16
//!     sig0 = (i64, i64) -> i8, i8 system_v
//!     fn0 = colocated u0:23 sig0 ; Instance { def: Item(WithOptConstParam { did: DefId(0:46 ~ example[4e51]::{impl#1}::call_mut), const_param_did: None }), substs: [ReErased, ReErased] }
//!
//! block0(v0: i64):
//!     nop
//! ; write_cvalue: Addr(Pointer { base: Stack(ss0), offset: Offset32(0) }, None): &&[u16] <- ByVal(v0): &&[u16]
//!     stack_store v0, ss0
//!     jump block1
//!
//! block1:
//!     nop
//! ; _3 = &mut _1
//!     v1 = iconst.i64 8
//! ; write_cvalue: Var(_3, var2): &mut IsNotEmpty <- ByVal(v1): &mut IsNotEmpty
//! ;
//! ; _0 = <IsNotEmpty as mini_core::FnMut<(&&[u16],)>>::call_mut(move _3, _2)
//!     v2 = stack_load.i64 ss0
//!     v3, v4 = call fn0(v1, v2)  ; v1 = 8
//!     v5 -> v3
//!     v6 -> v4
//! ; write_cvalue: VarPair(_0, var0, var1): (u8, u8) <- ByValPair(v3, v4): (u8, u8)
//!     jump block2
//!
//! block2:
//!     nop
//! ;
//! ; return
//!     return v5, v6
//! }
//! ```

use std::fmt;
use std::io::Write;

use cranelift_codegen::{
    entity::SecondaryMap,
    ir::entities::AnyEntity,
    write::{FuncWriter, PlainWriter},
};

use rustc_middle::ty::layout::FnAbiOf;
use rustc_session::config::{OutputFilenames, OutputType};

use crate::prelude::*;

#[derive(Clone, Debug)]
pub(crate) struct CommentWriter {
    enabled: bool,
    global_comments: Vec<String>,
    entity_comments: FxHashMap<AnyEntity, String>,
}

impl CommentWriter {
    pub(crate) fn new<'tcx>(tcx: TyCtxt<'tcx>, instance: Instance<'tcx>) -> Self {
        let enabled = should_write_ir(tcx);
        let global_comments = if enabled {
            vec![
                format!("symbol {}", tcx.symbol_name(instance).name),
                format!("instance {:?}", instance),
                format!(
                    "abi {:?}",
                    RevealAllLayoutCx(tcx).fn_abi_of_instance(instance, ty::List::empty())
                ),
                String::new(),
            ]
        } else {
            vec![]
        };

        CommentWriter { enabled, global_comments, entity_comments: FxHashMap::default() }
    }
}

impl CommentWriter {
    pub(crate) fn enabled(&self) -> bool {
        self.enabled
    }

    pub(crate) fn add_global_comment<S: Into<String>>(&mut self, comment: S) {
        debug_assert!(self.enabled);
        self.global_comments.push(comment.into());
    }

    pub(crate) fn add_comment<S: Into<String> + AsRef<str>, E: Into<AnyEntity>>(
        &mut self,
        entity: E,
        comment: S,
    ) {
        debug_assert!(self.enabled);

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
    ) -> Result<bool, fmt::Error> {
        for comment in &self.global_comments {
            if !comment.is_empty() {
                writeln!(w, "; {}", comment)?;
            } else {
                writeln!(w)?;
            }
        }
        if !self.global_comments.is_empty() {
            writeln!(w)?;
        }

        self.super_preamble(w, func)
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
            writeln!(w)
        }
    }

    fn write_block_header(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        block: Block,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_block_header(w, func, block, indent)
    }

    fn write_instruction(
        &mut self,
        w: &mut dyn fmt::Write,
        func: &Function,
        aliases: &SecondaryMap<Value, Vec<Value>>,
        inst: Inst,
        indent: usize,
    ) -> fmt::Result {
        PlainWriter.write_instruction(w, func, aliases, inst, indent)?;
        if let Some(comment) = self.entity_comments.get(&inst.into()) {
            writeln!(w, "; {}", comment.replace('\n', "\n; "))?;
        }
        Ok(())
    }
}

impl FunctionCx<'_, '_, '_> {
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

pub(crate) fn should_write_ir(tcx: TyCtxt<'_>) -> bool {
    tcx.sess.opts.output_types.contains_key(&OutputType::LlvmAssembly)
}

pub(crate) fn write_ir_file(
    output_filenames: &OutputFilenames,
    name: &str,
    write: impl FnOnce(&mut dyn Write) -> std::io::Result<()>,
) {
    let clif_output_dir = output_filenames.with_extension("clif");

    match std::fs::create_dir(&clif_output_dir) {
        Ok(()) => {}
        Err(err) if err.kind() == std::io::ErrorKind::AlreadyExists => {}
        res @ Err(_) => res.unwrap(),
    }

    let clif_file_name = clif_output_dir.join(name);

    let res = std::fs::File::create(clif_file_name).and_then(|mut file| write(&mut file));
    if let Err(err) = res {
        // Using early_warn as no Session is available here
        let handler = rustc_session::EarlyErrorHandler::new(
            rustc_session::config::ErrorOutputType::default(),
        );
        handler.early_warn(format!("error writing ir file: {}", err));
    }
}

pub(crate) fn write_clif_file(
    output_filenames: &OutputFilenames,
    symbol_name: &str,
    postfix: &str,
    isa: &dyn cranelift_codegen::isa::TargetIsa,
    func: &cranelift_codegen::ir::Function,
    mut clif_comments: &CommentWriter,
) {
    // FIXME work around filename too long errors
    write_ir_file(output_filenames, &format!("{}.{}.clif", symbol_name, postfix), |file| {
        let mut clif = String::new();
        cranelift_codegen::write::decorate_function(&mut clif_comments, &mut clif, func).unwrap();

        for flag in isa.flags().iter() {
            writeln!(file, "set {}", flag)?;
        }
        write!(file, "target {}", isa.triple().architecture)?;
        for isa_flag in isa.isa_flags().iter() {
            write!(file, " {}", isa_flag)?;
        }
        writeln!(file, "\n")?;
        writeln!(file)?;
        file.write_all(clif.as_bytes())?;
        Ok(())
    });
}

impl fmt::Debug for FunctionCx<'_, '_, '_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "{:?}", self.instance.substs)?;
        writeln!(f, "{:?}", self.local_map)?;

        let mut clif = String::new();
        ::cranelift_codegen::write::decorate_function(
            &mut &self.clif_comments,
            &mut clif,
            &self.bcx.func,
        )
        .unwrap();
        writeln!(f, "\n{}", clif)
    }
}

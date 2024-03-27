//! Handling of everything related to debuginfo.

mod emit;
mod line_info;
mod object;
mod unwind;

use cranelift_codegen::ir::Endianness;
use cranelift_codegen::isa::TargetIsa;
use gimli::write::{
    Address, AttributeValue, DwarfUnit, Expression, FileId, LineProgram, LineString, Range,
    RangeList, UnitEntryId,
};
use gimli::{AArch64, Encoding, Format, LineEncoding, Register, RiscV, RunTimeEndian, X86_64};
use indexmap::IndexSet;
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_hir::def_id::DefIdMap;
use rustc_session::Session;

pub(crate) use self::emit::{DebugReloc, DebugRelocName};
pub(crate) use self::unwind::UnwindContext;
use crate::prelude::*;

pub(crate) fn producer(sess: &Session) -> String {
    format!("rustc version {} with cranelift {}", sess.cfg_version, cranelift_codegen::VERSION)
}

pub(crate) struct DebugContext {
    endian: RunTimeEndian,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,
    stack_pointer_register: Register,
    namespace_map: DefIdMap<UnitEntryId>,

    should_remap_filepaths: bool,
}

pub(crate) struct FunctionDebugContext {
    entry_id: UnitEntryId,
    function_source_loc: (FileId, u64, u64),
    source_loc_set: IndexSet<(FileId, u64, u64)>,
}

impl DebugContext {
    pub(crate) fn new(tcx: TyCtxt<'_>, isa: &dyn TargetIsa) -> Self {
        let encoding = Encoding {
            format: Format::Dwarf32,
            // FIXME this should be configurable
            // macOS doesn't seem to support DWARF > 3
            // 5 version is required for md5 file hash
            version: if tcx.sess.target.is_like_osx {
                3
            } else {
                // FIXME change to version 5 once the gdb and lldb shipping with the latest debian
                // support it.
                4
            },
            address_size: isa.frontend_config().pointer_bytes(),
        };

        let endian = match isa.endianness() {
            Endianness::Little => RunTimeEndian::Little,
            Endianness::Big => RunTimeEndian::Big,
        };

        let stack_pointer_register = match isa.triple().architecture {
            target_lexicon::Architecture::Aarch64(_) => AArch64::SP,
            target_lexicon::Architecture::Riscv64(_) => RiscV::SP,
            target_lexicon::Architecture::X86_64 | target_lexicon::Architecture::X86_64h => {
                X86_64::RSP
            }
            _ => Register(u16::MAX),
        };

        let mut dwarf = DwarfUnit::new(encoding);

        let should_remap_filepaths = tcx.sess.should_prefer_remapped_for_codegen();

        let producer = producer(tcx.sess);
        let comp_dir = tcx
            .sess
            .opts
            .working_dir
            .to_string_lossy(if should_remap_filepaths {
                FileNameDisplayPreference::Remapped
            } else {
                FileNameDisplayPreference::Local
            })
            .into_owned();
        let (name, file_info) = match tcx.sess.local_crate_source_file() {
            Some(path) => {
                let name = path.to_string_lossy().into_owned();
                (name, None)
            }
            None => (tcx.crate_name(LOCAL_CRATE).to_string(), None),
        };

        let mut line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::new(comp_dir.as_bytes(), encoding, &mut dwarf.line_strings),
            LineString::new(name.as_bytes(), encoding, &mut dwarf.line_strings),
            file_info,
        );
        line_program.file_has_md5 = file_info.is_some();

        dwarf.unit.line_program = line_program;

        {
            let name = dwarf.strings.add(name);
            let comp_dir = dwarf.strings.add(comp_dir);

            let root = dwarf.unit.root();
            let root = dwarf.unit.get_mut(root);
            root.set(gimli::DW_AT_producer, AttributeValue::StringRef(dwarf.strings.add(producer)));
            root.set(gimli::DW_AT_language, AttributeValue::Language(gimli::DW_LANG_Rust));
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name));
            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir));
            root.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));
        }

        DebugContext {
            endian,
            dwarf,
            unit_range_list: RangeList(Vec::new()),
            stack_pointer_register,
            namespace_map: DefIdMap::default(),
            should_remap_filepaths,
        }
    }

    fn item_namespace(&mut self, tcx: TyCtxt<'_>, def_id: DefId) -> UnitEntryId {
        if let Some(&scope) = self.namespace_map.get(&def_id) {
            return scope;
        }

        let def_key = tcx.def_key(def_id);
        let parent_scope = def_key
            .parent
            .map(|parent| self.item_namespace(tcx, DefId { krate: def_id.krate, index: parent }))
            .unwrap_or(self.dwarf.unit.root());

        let namespace_name = {
            let mut output = String::new();
            type_names::push_item_name(tcx, def_id, false, &mut output);
            output
        };
        let namespace_name_id = self.dwarf.strings.add(namespace_name);

        let scope = self.dwarf.unit.add(parent_scope, gimli::DW_TAG_namespace);
        let scope_entry = self.dwarf.unit.get_mut(scope);
        scope_entry.set(gimli::DW_AT_name, AttributeValue::StringRef(namespace_name_id));

        self.namespace_map.insert(def_id, scope);
        scope
    }

    pub(crate) fn define_function<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        linkage_name: &str,
        function_span: Span,
    ) -> FunctionDebugContext {
        let (file, line, column) = DebugContext::get_span_loc(tcx, function_span, function_span);

        let file_id = self.add_source_file(&file);

        // FIXME: add to appropriate scope instead of root
        let scope = self.item_namespace(tcx, tcx.parent(instance.def_id()));

        let mut name = String::new();
        type_names::push_item_name(tcx, instance.def_id(), false, &mut name);

        // Find the enclosing function, in case this is a closure.
        let enclosing_fn_def_id = tcx.typeck_root_def_id(instance.def_id());

        // We look up the generics of the enclosing function and truncate the args
        // to their length in order to cut off extra stuff that might be in there for
        // closures or coroutines.
        let generics = tcx.generics_of(enclosing_fn_def_id);
        let args = instance.args.truncate_to(tcx, generics);

        type_names::push_generic_params(
            tcx,
            tcx.normalize_erasing_regions(ty::ParamEnv::reveal_all(), args),
            enclosing_fn_def_id,
            &mut name,
        );

        let entry_id = self.dwarf.unit.add(scope, gimli::DW_TAG_subprogram);
        let entry = self.dwarf.unit.get_mut(entry_id);
        let linkage_name_id =
            if name != linkage_name { Some(self.dwarf.strings.add(linkage_name)) } else { None };
        let name_id = self.dwarf.strings.add(name);

        // These will be replaced in FunctionDebugContext::finalize. They are
        // only defined here to ensure that the order of the attributes matches
        // rustc.
        entry.set(gimli::DW_AT_low_pc, AttributeValue::Udata(0));
        entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(0));

        let mut frame_base_expr = Expression::new();
        frame_base_expr.op_reg(self.stack_pointer_register);
        entry.set(gimli::DW_AT_frame_base, AttributeValue::Exprloc(frame_base_expr));

        if let Some(linkage_name_id) = linkage_name_id {
            entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(linkage_name_id));
        }
        // Gdb requires DW_AT_name. Otherwise the DW_TAG_subprogram is skipped.
        // FIXME only include the function name and not the full mangled symbol
        entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));

        entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
        entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(line));

        // FIXME set DW_AT_external as appropriate

        FunctionDebugContext {
            entry_id,
            function_source_loc: (file_id, line, column),
            source_loc_set: IndexSet::new(),
        }
    }
}

impl FunctionDebugContext {
    pub(crate) fn finalize(
        mut self,
        debug_context: &mut DebugContext,
        func_id: FuncId,
        context: &Context,
    ) {
        let symbol = func_id.as_u32() as usize;

        let end = self.create_debug_lines(debug_context, symbol, context);

        debug_context.unit_range_list.0.push(Range::StartLength {
            begin: Address::Symbol { symbol, addend: 0 },
            length: u64::from(end),
        });

        let func_entry = debug_context.dwarf.unit.get_mut(self.entry_id);
        // Gdb requires both DW_AT_low_pc and DW_AT_high_pc. Otherwise the DW_TAG_subprogram is skipped.
        func_entry.set(
            gimli::DW_AT_low_pc,
            AttributeValue::Address(Address::Symbol { symbol, addend: 0 }),
        );
        // Using Udata for DW_AT_high_pc requires at least DWARF4
        func_entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(u64::from(end)));
    }
}

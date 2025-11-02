//! Handling of everything related to debuginfo.

mod emit;
mod line_info;
mod object;
mod types;
mod unwind;

use cranelift_codegen::ir::Endianness;
use cranelift_codegen::isa::TargetIsa;
use cranelift_module::DataId;
use gimli::write::{
    Address, AttributeValue, DwarfUnit, Expression, FileId, LineProgram, LineString, Range,
    RangeList, UnitEntryId,
};
use gimli::{AArch64, Encoding, Format, LineEncoding, Register, RiscV, RunTimeEndian, X86_64};
use indexmap::IndexSet;
use rustc_codegen_ssa::debuginfo::type_names;
use rustc_hir::def::DefKind;
use rustc_hir::def_id::DefIdMap;
use rustc_session::Session;
use rustc_span::{FileNameDisplayPreference, SourceFileHash, StableSourceFileId};
use rustc_target::callconv::FnAbi;

pub(crate) use self::emit::{DebugReloc, DebugRelocName};
pub(crate) use self::types::TypeDebugContext;
pub(crate) use self::unwind::UnwindContext;
use crate::debuginfo::emit::{address_for_data, address_for_func};
use crate::prelude::*;

pub(crate) fn producer(sess: &Session) -> String {
    format!("rustc version {} with cranelift {}", sess.cfg_version, cranelift_codegen::VERSION)
}

pub(crate) struct DebugContext {
    endian: RunTimeEndian,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,
    created_files: FxHashMap<(StableSourceFileId, SourceFileHash), FileId>,
    stack_pointer_register: Register,
    namespace_map: DefIdMap<UnitEntryId>,
    array_size_type: UnitEntryId,

    filename_display_preference: FileNameDisplayPreference,
}

pub(crate) struct FunctionDebugContext {
    entry_id: UnitEntryId,
    function_source_loc: (FileId, u64, u64),
    source_loc_set: IndexSet<(FileId, u64, u64)>,
}

impl DebugContext {
    pub(crate) fn new(tcx: TyCtxt<'_>, isa: &dyn TargetIsa, cgu_name: &str) -> Self {
        let encoding = Encoding {
            format: Format::Dwarf32,
            // FIXME this should be configurable
            // macOS doesn't seem to support DWARF > 3
            // 5 version is required for md5 file hash
            version: if tcx.sess.target.is_like_darwin {
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

        use rustc_session::config::RemapPathScopeComponents;

        let filename_display_preference =
            tcx.sess.filename_display_preference(RemapPathScopeComponents::DEBUGINFO);

        let producer = producer(tcx.sess);
        let comp_dir =
            tcx.sess.opts.working_dir.to_string_lossy(filename_display_preference).to_string();

        let (name, file_info) = match tcx.sess.local_crate_source_file() {
            Some(path) => {
                let name = path.to_string_lossy(filename_display_preference).to_string();
                (name, None)
            }
            None => (tcx.crate_name(LOCAL_CRATE).to_string(), None),
        };

        let file_has_md5 = file_info.is_some();
        let mut line_program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::new(comp_dir.as_bytes(), encoding, &mut dwarf.line_strings),
            LineString::new(name.as_bytes(), encoding, &mut dwarf.line_strings),
            file_info,
        );
        line_program.file_has_md5 = file_has_md5;

        dwarf.unit.line_program = line_program;

        {
            let name = dwarf.strings.add(format!("{name}/@/{cgu_name}"));
            let comp_dir = dwarf.strings.add(comp_dir);

            let root = dwarf.unit.root();
            let root = dwarf.unit.get_mut(root);
            root.set(gimli::DW_AT_producer, AttributeValue::StringRef(dwarf.strings.add(producer)));
            root.set(gimli::DW_AT_language, AttributeValue::Language(gimli::DW_LANG_Rust));
            root.set(gimli::DW_AT_name, AttributeValue::StringRef(name));

            // This will be replaced when emitting the debuginfo. It is only
            // defined here to ensure that the order of the attributes matches
            // rustc.
            root.set(gimli::DW_AT_stmt_list, AttributeValue::Udata(0));

            root.set(gimli::DW_AT_comp_dir, AttributeValue::StringRef(comp_dir));
            root.set(gimli::DW_AT_low_pc, AttributeValue::Address(Address::Constant(0)));
        }

        let array_size_type = dwarf.unit.add(dwarf.unit.root(), gimli::DW_TAG_base_type);
        let array_size_type_entry = dwarf.unit.get_mut(array_size_type);
        array_size_type_entry.set(
            gimli::DW_AT_name,
            AttributeValue::StringRef(dwarf.strings.add("__ARRAY_SIZE_TYPE__")),
        );
        array_size_type_entry
            .set(gimli::DW_AT_encoding, AttributeValue::Encoding(gimli::DW_ATE_unsigned));
        array_size_type_entry.set(
            gimli::DW_AT_byte_size,
            AttributeValue::Udata(isa.frontend_config().pointer_bytes().into()),
        );

        DebugContext {
            endian,
            dwarf,
            unit_range_list: RangeList(Vec::new()),
            created_files: FxHashMap::default(),
            stack_pointer_register,
            namespace_map: DefIdMap::default(),
            array_size_type,
            filename_display_preference,
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
        type_dbg: &mut TypeDebugContext<'tcx>,
        instance: Instance<'tcx>,
        fn_abi: &'tcx FnAbi<'tcx, Ty<'tcx>>,
        linkage_name: &str,
        function_span: Span,
    ) -> FunctionDebugContext {
        let (file_id, line, column) = self.get_span_loc(tcx, function_span, function_span);

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
            tcx.normalize_erasing_regions(ty::TypingEnv::fully_monomorphized(), args),
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
        entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));

        entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
        entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(line));

        if !fn_abi.ret.is_ignore() {
            let return_dw_ty = self.debug_type(tcx, type_dbg, fn_abi.ret.layout.ty);
            let entry = self.dwarf.unit.get_mut(entry_id);
            entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(return_dw_ty));
        }

        if tcx.is_reachable_non_generic(instance.def_id()) {
            let entry = self.dwarf.unit.get_mut(entry_id);
            entry.set(gimli::DW_AT_external, AttributeValue::FlagPresent);
        }

        FunctionDebugContext {
            entry_id,
            function_source_loc: (file_id, line, column),
            source_loc_set: IndexSet::new(),
        }
    }

    // Adapted from https://github.com/rust-lang/rust/blob/10a7aa14fed9b528b74b0f098c4899c37c09a9c7/compiler/rustc_codegen_llvm/src/debuginfo/metadata.rs#L1288-L1346
    pub(crate) fn define_static<'tcx>(
        &mut self,
        tcx: TyCtxt<'tcx>,
        type_dbg: &mut TypeDebugContext<'tcx>,
        def_id: DefId,
        data_id: DataId,
    ) {
        let DefKind::Static { nested, .. } = tcx.def_kind(def_id) else { bug!() };
        if nested {
            return;
        }

        let scope = self.item_namespace(tcx, tcx.parent(def_id));

        let span = tcx.def_span(def_id);
        let (file_id, line, _column) = self.get_span_loc(tcx, span, span);

        let static_type = Instance::mono(tcx, def_id).ty(tcx, ty::TypingEnv::fully_monomorphized());
        let static_layout = tcx
            .layout_of(ty::TypingEnv::fully_monomorphized().as_query_input(static_type))
            .unwrap();
        // FIXME use the actual type layout
        let type_id = self.debug_type(tcx, type_dbg, static_type);

        let name = tcx.item_name(def_id);
        let linkage_name = tcx.symbol_name(Instance::mono(tcx, def_id)).name;

        let entry_id = self.dwarf.unit.add(scope, gimli::DW_TAG_variable);
        let entry = self.dwarf.unit.get_mut(entry_id);
        let linkage_name_id = if name.as_str() != linkage_name {
            Some(self.dwarf.strings.add(linkage_name))
        } else {
            None
        };
        let name_id = self.dwarf.strings.add(name.as_str());

        entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
        entry.set(gimli::DW_AT_type, AttributeValue::UnitRef(type_id));

        if tcx.is_reachable_non_generic(def_id) {
            entry.set(gimli::DW_AT_external, AttributeValue::FlagPresent);
        }

        entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
        entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(line));

        entry.set(gimli::DW_AT_alignment, AttributeValue::Udata(static_layout.align.bytes()));

        let mut expr = Expression::new();
        expr.op_addr(address_for_data(data_id));
        entry.set(gimli::DW_AT_location, AttributeValue::Exprloc(expr));

        if let Some(linkage_name_id) = linkage_name_id {
            entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(linkage_name_id));
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
        let end = self.create_debug_lines(debug_context, func_id, context);

        debug_context
            .unit_range_list
            .0
            .push(Range::StartLength { begin: address_for_func(func_id), length: u64::from(end) });

        let func_entry = debug_context.dwarf.unit.get_mut(self.entry_id);
        // Gdb requires both DW_AT_low_pc and DW_AT_high_pc. Otherwise the DW_TAG_subprogram is skipped.
        func_entry.set(gimli::DW_AT_low_pc, AttributeValue::Address(address_for_func(func_id)));
        // Using Udata for DW_AT_high_pc requires at least DWARF4
        func_entry.set(gimli::DW_AT_high_pc, AttributeValue::Udata(u64::from(end)));
    }
}

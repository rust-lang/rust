//! Handling of everything related to debuginfo.

mod emit;
mod line_info;
mod object;
mod unwind;

use crate::prelude::*;

use cranelift_codegen::ir::Endianness;
use cranelift_codegen::isa::TargetIsa;

use gimli::write::{
    Address, AttributeValue, DwarfUnit, FileId, LineProgram, LineString, Range, RangeList,
    UnitEntryId,
};
use gimli::{Encoding, Format, LineEncoding, RunTimeEndian};
use indexmap::IndexSet;

pub(crate) use emit::{DebugReloc, DebugRelocName};
pub(crate) use unwind::UnwindContext;

pub(crate) struct DebugContext {
    endian: RunTimeEndian,

    dwarf: DwarfUnit,
    unit_range_list: RangeList,
}

pub(crate) struct FunctionDebugContext {
    entry_id: UnitEntryId,
    function_source_loc: (FileId, u64, u64),
    source_loc_set: indexmap::IndexSet<(FileId, u64, u64)>,
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

        let mut dwarf = DwarfUnit::new(encoding);

        let producer = format!(
            "cg_clif (rustc {}, cranelift {})",
            rustc_interface::util::version_str().unwrap_or("unknown version"),
            cranelift_codegen::VERSION,
        );
        let comp_dir = tcx
            .sess
            .opts
            .working_dir
            .to_string_lossy(FileNameDisplayPreference::Remapped)
            .into_owned();
        let (name, file_info) = match tcx.sess.local_crate_source_file.clone() {
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

        DebugContext { endian, dwarf, unit_range_list: RangeList(Vec::new()) }
    }

    pub(crate) fn define_function(
        &mut self,
        tcx: TyCtxt<'_>,
        name: &str,
        function_span: Span,
    ) -> FunctionDebugContext {
        let (file, line, column) = DebugContext::get_span_loc(tcx, function_span, function_span);

        let file_id = self.add_source_file(&file);

        // FIXME: add to appropriate scope instead of root
        let scope = self.dwarf.unit.root();

        let entry_id = self.dwarf.unit.add(scope, gimli::DW_TAG_subprogram);
        let entry = self.dwarf.unit.get_mut(entry_id);
        let name_id = self.dwarf.strings.add(name);
        // Gdb requires DW_AT_name. Otherwise the DW_TAG_subprogram is skipped.
        entry.set(gimli::DW_AT_name, AttributeValue::StringRef(name_id));
        entry.set(gimli::DW_AT_linkage_name, AttributeValue::StringRef(name_id));

        entry.set(gimli::DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
        entry.set(gimli::DW_AT_decl_line, AttributeValue::Udata(line));
        entry.set(gimli::DW_AT_decl_column, AttributeValue::Udata(column));

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

//! Errors emitted by codegen_ssa

use crate::back::command::Command;
use rustc_errors::{
    fluent, DiagnosticArgValue, DiagnosticBuilder, ErrorGuaranteed, Handler, IntoDiagnostic,
    IntoDiagnosticArg,
};
use rustc_macros::Diagnostic;
use rustc_span::{Span, Symbol};
use std::borrow::Cow;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

#[derive(Diagnostic)]
#[diag(codegen_ssa_lib_def_write_failure)]
pub struct LibDefWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_version_script_write_failure)]
pub struct VersionScriptWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_symbol_file_write_failure)]
pub struct SymbolFileWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_ld64_unimplemented_modifier)]
pub struct Ld64UnimplementedModifier;

#[derive(Diagnostic)]
#[diag(codegen_ssa_linker_unsupported_modifier)]
pub struct LinkerUnsupportedModifier;

#[derive(Diagnostic)]
#[diag(codegen_ssa_L4Bender_exporting_symbols_unimplemented)]
pub struct L4BenderExportingSymbolsUnimplemented;

#[derive(Diagnostic)]
#[diag(codegen_ssa_no_natvis_directory)]
pub struct NoNatvisDirectory {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_copy_path_buf)]
pub struct CopyPathBuf {
    pub source_file: PathBuf,
    pub output_path: PathBuf,
    pub error: Error,
}

// Reports Paths using `Debug` implementation rather than Path's `Display` implementation.
#[derive(Diagnostic)]
#[diag(codegen_ssa_copy_path)]
pub struct CopyPath<'a> {
    from: DebugArgPath<'a>,
    to: DebugArgPath<'a>,
    error: Error,
}

impl<'a> CopyPath<'a> {
    pub fn new(from: &'a Path, to: &'a Path, error: Error) -> CopyPath<'a> {
        CopyPath { from: DebugArgPath(from), to: DebugArgPath(to), error }
    }
}

struct DebugArgPath<'a>(pub &'a Path);

impl IntoDiagnosticArg for DebugArgPath<'_> {
    fn into_diagnostic_arg(self) -> rustc_errors::DiagnosticArgValue<'static> {
        DiagnosticArgValue::Str(Cow::Owned(format!("{:?}", self.0)))
    }
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_ignoring_emit_path)]
pub struct IgnoringEmitPath {
    pub extension: String,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_ignoring_output)]
pub struct IgnoringOutput {
    pub extension: String,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_create_temp_dir)]
pub struct CreateTempDir {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_incompatible_linking_modifiers)]
pub struct IncompatibleLinkingModifiers;

#[derive(Diagnostic)]
#[diag(codegen_ssa_add_native_library)]
pub struct AddNativeLibrary {
    pub library_path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(codegen_ssa_multiple_external_func_decl)]
pub struct MultipleExternalFuncDecl<'a> {
    #[primary_span]
    pub span: Span,
    pub function: Symbol,
    pub library_name: &'a str,
}

#[derive(Diagnostic)]
pub enum LinkRlibError {
    #[diag(codegen_ssa_rlib_missing_format)]
    MissingFormat,

    #[diag(codegen_ssa_rlib_only_rmeta_found)]
    OnlyRmetaFound { crate_name: Symbol },

    #[diag(codegen_ssa_rlib_not_found)]
    NotFound { crate_name: Symbol },

    #[diag(codegen_ssa_rlib_incompatible_dependency_formats)]
    IncompatibleDependencyFormats { ty1: String, ty2: String, list1: String, list2: String },
}

pub struct ThorinErrorWrapper(pub thorin::Error);

impl IntoDiagnostic<'_> for ThorinErrorWrapper {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag;
        match self.0 {
            thorin::Error::ReadInput(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_read_input_failure);
                diag
            }
            thorin::Error::ParseFileKind(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_input_file_kind);
                diag
            }
            thorin::Error::ParseObjectFile(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_input_object_file);
                diag
            }
            thorin::Error::ParseArchiveFile(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_input_archive_file);
                diag
            }
            thorin::Error::ParseArchiveMember(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_archive_member);
                diag
            }
            thorin::Error::InvalidInputKind => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_invalid_input_kind);
                diag
            }
            thorin::Error::DecompressData(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_decompress_data);
                diag
            }
            thorin::Error::NamelessSection(_, offset) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_section_without_name);
                diag.set_arg("offset", format!("0x{:08x}", offset));
                diag
            }
            thorin::Error::RelocationWithInvalidSymbol(section, offset) => {
                diag =
                    handler.struct_err(fluent::codegen_ssa_thorin_relocation_with_invalid_symbol);
                diag.set_arg("section", section);
                diag.set_arg("offset", format!("0x{:08x}", offset));
                diag
            }
            thorin::Error::MultipleRelocations(section, offset) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_multiple_relocations);
                diag.set_arg("section", section);
                diag.set_arg("offset", format!("0x{:08x}", offset));
                diag
            }
            thorin::Error::UnsupportedRelocation(section, offset) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_unsupported_relocation);
                diag.set_arg("section", section);
                diag.set_arg("offset", format!("0x{:08x}", offset));
                diag
            }
            thorin::Error::MissingDwoName(id) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_missing_dwo_name);
                diag.set_arg("id", format!("0x{:08x}", id));
                diag
            }
            thorin::Error::NoCompilationUnits => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_no_compilation_units);
                diag
            }
            thorin::Error::NoDie => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_no_die);
                diag
            }
            thorin::Error::TopLevelDieNotUnit => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_top_level_die_not_unit);
                diag
            }
            thorin::Error::MissingRequiredSection(section) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_missing_required_section);
                diag.set_arg("section", section);
                diag
            }
            thorin::Error::ParseUnitAbbreviations(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_unit_abbreviations);
                diag
            }
            thorin::Error::ParseUnitAttribute(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_unit_attribute);
                diag
            }
            thorin::Error::ParseUnitHeader(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_unit_header);
                diag
            }
            thorin::Error::ParseUnit(_) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_unit);
                diag
            }
            thorin::Error::IncompatibleIndexVersion(section, format, actual) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_incompatible_index_version);
                diag.set_arg("section", section);
                diag.set_arg("actual", actual);
                diag.set_arg("format", format);
                diag
            }
            thorin::Error::OffsetAtIndex(_, index) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_offset_at_index);
                diag.set_arg("index", index);
                diag
            }
            thorin::Error::StrAtOffset(_, offset) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_str_at_offset);
                diag.set_arg("offset", format!("0x{:08x}", offset));
                diag
            }
            thorin::Error::ParseIndex(_, section) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_parse_index);
                diag.set_arg("section", section);
                diag
            }
            thorin::Error::UnitNotInIndex(unit) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_unit_not_in_index);
                diag.set_arg("unit", format!("0x{:08x}", unit));
                diag
            }
            thorin::Error::RowNotInIndex(_, row) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_row_not_in_index);
                diag.set_arg("row", row);
                diag
            }
            thorin::Error::SectionNotInRow => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_section_not_in_row);
                diag
            }
            thorin::Error::EmptyUnit(unit) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_empty_unit);
                diag.set_arg("unit", format!("0x{:08x}", unit));
                diag
            }
            thorin::Error::MultipleDebugInfoSection => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_multiple_debug_info_section);
                diag
            }
            thorin::Error::MultipleDebugTypesSection => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_multiple_debug_types_section);
                diag
            }
            thorin::Error::NotSplitUnit => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_not_split_unit);
                diag
            }
            thorin::Error::DuplicateUnit(unit) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_duplicate_unit);
                diag.set_arg("unit", format!("0x{:08x}", unit));
                diag
            }
            thorin::Error::MissingReferencedUnit(unit) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_missing_referenced_unit);
                diag.set_arg("unit", format!("0x{:08x}", unit));
                diag
            }
            thorin::Error::NoOutputObjectCreated => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_not_output_object_created);
                diag
            }
            thorin::Error::MixedInputEncodings => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_mixed_input_encodings);
                diag
            }
            thorin::Error::Io(e) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_io);
                diag.set_arg("error", format!("{e}"));
                diag
            }
            thorin::Error::ObjectRead(e) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_object_read);
                diag.set_arg("error", format!("{e}"));
                diag
            }
            thorin::Error::ObjectWrite(e) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_object_write);
                diag.set_arg("error", format!("{e}"));
                diag
            }
            thorin::Error::GimliRead(e) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_gimli_read);
                diag.set_arg("error", format!("{e}"));
                diag
            }
            thorin::Error::GimliWrite(e) => {
                diag = handler.struct_err(fluent::codegen_ssa_thorin_gimli_write);
                diag.set_arg("error", format!("{e}"));
                diag
            }
            _ => unimplemented!("Untranslated thorin error"),
        }
    }
}

pub struct LinkingFailed<'a> {
    pub linker_path: &'a PathBuf,
    pub exit_status: ExitStatus,
    pub command: &'a Command,
    pub escaped_output: &'a str,
}

impl IntoDiagnostic<'_> for LinkingFailed<'_> {
    fn into_diagnostic(self, handler: &Handler) -> DiagnosticBuilder<'_, ErrorGuaranteed> {
        let mut diag = handler.struct_err(fluent::codegen_ssa_linking_failed);
        diag.set_arg("linker_path", format!("{}", self.linker_path.display()));
        diag.set_arg("exit_status", format!("{}", self.exit_status));

        diag.note(format!("{:?}", self.command)).note(self.escaped_output);

        // Trying to match an error from OS linkers
        // which by now we have no way to translate.
        if self.escaped_output.contains("undefined reference to") {
            diag.note(fluent::codegen_ssa_extern_funcs_not_found)
                .note(fluent::codegen_ssa_specify_libraries_to_link)
                .note(fluent::codegen_ssa_use_cargo_directive);
        }
        diag
    }
}

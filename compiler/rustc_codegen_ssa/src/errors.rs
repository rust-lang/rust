//! Errors emitted by codegen_ssa

use std::borrow::Cow;
use std::ffi::OsString;
use std::io::Error;
use std::path::{Path, PathBuf};
use std::process::ExitStatus;

use rustc_errors::codes::*;
use rustc_errors::{
    Diag, DiagArgValue, DiagCtxtHandle, Diagnostic, EmissionGuarantee, IntoDiagArg, Level,
    inline_fluent,
};
use rustc_macros::{Diagnostic, LintDiagnostic, Subdiagnostic};
use rustc_middle::ty::layout::LayoutError;
use rustc_middle::ty::{FloatTy, Ty};
use rustc_span::{Span, Symbol};

use crate::assert_module_sources::CguReuse;
use crate::back::command::Command;

#[derive(Diagnostic)]
#[diag(
    "CGU-reuse for `{$cgu_user_name}` is `{$actual_reuse}` but should be {$at_least ->
        [one] {\"at least \"}
        *[other] {\"\"}
    }`{$expected_reuse}`"
)]
pub(crate) struct IncorrectCguReuseType<'a> {
    #[primary_span]
    pub span: Span,
    pub cgu_user_name: &'a str,
    pub actual_reuse: CguReuse,
    pub expected_reuse: CguReuse,
    pub at_least: u8,
}

#[derive(Diagnostic)]
#[diag("CGU-reuse for `{$cgu_user_name}` is (mangled: `{$cgu_name}`) was not recorded")]
pub(crate) struct CguNotRecorded<'a> {
    pub cgu_user_name: &'a str,
    pub cgu_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("found CGU-reuse attribute but `-Zquery-dep-graph` was not specified")]
pub(crate) struct MissingQueryDepGraph {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag(
    "found malformed codegen unit name `{$user_path}`. codegen units names must always start with the name of the crate (`{$crate_name}` in this case)"
)]
pub(crate) struct MalformedCguName<'a> {
    #[primary_span]
    pub span: Span,
    pub user_path: &'a str,
    pub crate_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("no module named `{$user_path}` (mangled: {$cgu_name}). available modules: {$cgu_names}")]
pub(crate) struct NoModuleNamed<'a> {
    #[primary_span]
    pub span: Span,
    pub user_path: &'a str,
    pub cgu_name: Symbol,
    pub cgu_names: String,
}

#[derive(Diagnostic)]
#[diag("failed to write lib.def file: {$error}")]
pub(crate) struct LibDefWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("failed to write version script: {$error}")]
pub(crate) struct VersionScriptWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("failed to write symbols file: {$error}")]
pub(crate) struct SymbolFileWriteFailure {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("`as-needed` modifier not implemented yet for ld64")]
pub(crate) struct Ld64UnimplementedModifier;

#[derive(Diagnostic)]
#[diag("`as-needed` modifier not supported for current linker")]
pub(crate) struct LinkerUnsupportedModifier;

#[derive(Diagnostic)]
#[diag("exporting symbols not implemented yet for L4Bender")]
pub(crate) struct L4BenderExportingSymbolsUnimplemented;

#[derive(Diagnostic)]
#[diag("error enumerating natvis directory: {$error}")]
pub(crate) struct NoNatvisDirectory {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("cached cgu {$cgu_name} should have an object file, but doesn't")]
pub(crate) struct NoSavedObjectFile<'a> {
    pub cgu_name: &'a str,
}

#[derive(Diagnostic)]
#[diag("`#[track_caller]` requires Rust ABI", code = E0737)]
pub(crate) struct RequiresRustAbi {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("unable to copy {$source_file} to {$output_path}: {$error}")]
pub(crate) struct CopyPathBuf {
    pub source_file: PathBuf,
    pub output_path: PathBuf,
    pub error: Error,
}

// Reports Paths using `Debug` implementation rather than Path's `Display` implementation.
#[derive(Diagnostic)]
#[diag("could not copy {$from} to {$to}: {$error}")]
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

impl IntoDiagArg for DebugArgPath<'_> {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> rustc_errors::DiagArgValue {
        DiagArgValue::Str(Cow::Owned(format!("{:?}", self.0)))
    }
}

#[derive(Diagnostic)]
#[diag(
    "option `-o` or `--emit` is used to write binary output type `{$shorthand}` to stdout, but stdout is a tty"
)]
pub struct BinaryOutputToTty {
    pub shorthand: &'static str,
}

#[derive(Diagnostic)]
#[diag("ignoring emit path because multiple .{$extension} files were produced")]
pub struct IgnoringEmitPath {
    pub extension: &'static str,
}

#[derive(Diagnostic)]
#[diag("ignoring -o because multiple .{$extension} files were produced")]
pub struct IgnoringOutput {
    pub extension: &'static str,
}

#[derive(Diagnostic)]
#[diag("couldn't create a temp dir: {$error}")]
pub(crate) struct CreateTempDir {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("failed to add native library {$library_path}: {$error}")]
pub(crate) struct AddNativeLibrary {
    pub library_path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag(
    "multiple declarations of external function `{$function}` from library `{$library_name}` have different calling conventions"
)]
pub(crate) struct MultipleExternalFuncDecl<'a> {
    #[primary_span]
    pub span: Span,
    pub function: Symbol,
    pub library_name: &'a str,
}

#[derive(Diagnostic)]
pub enum LinkRlibError {
    #[diag("could not find formats for rlibs")]
    MissingFormat,

    #[diag("could not find rlib for: `{$crate_name}`, found rmeta (metadata) file")]
    OnlyRmetaFound { crate_name: Symbol },

    #[diag("could not find rlib for: `{$crate_name}`")]
    NotFound { crate_name: Symbol },

    #[diag(
        "`{$ty1}` and `{$ty2}` do not have equivalent dependency formats (`{$list1}` vs `{$list2}`)"
    )]
    IncompatibleDependencyFormats { ty1: String, ty2: String, list1: String, list2: String },
}

pub(crate) struct ThorinErrorWrapper(pub thorin::Error);

impl<G: EmissionGuarantee> Diagnostic<'_, G> for ThorinErrorWrapper {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let build = |msg| Diag::new(dcx, level, msg);
        match self.0 {
            thorin::Error::ReadInput(_) => build(inline_fluent!("failed to read input file")),
            thorin::Error::ParseFileKind(_) => {
                build(inline_fluent!("failed to parse input file kind"))
            }
            thorin::Error::ParseObjectFile(_) => {
                build(inline_fluent!("failed to parse input object file"))
            }
            thorin::Error::ParseArchiveFile(_) => {
                build(inline_fluent!("failed to parse input archive file"))
            }
            thorin::Error::ParseArchiveMember(_) => {
                build(inline_fluent!("failed to parse archive member"))
            }
            thorin::Error::InvalidInputKind => build(inline_fluent!("input is not an archive or elf object")),
            thorin::Error::DecompressData(_) => build(inline_fluent!("failed to decompress compressed section")),
            thorin::Error::NamelessSection(_, offset) => {
                build(inline_fluent!("section without name at offset {$offset}"))
                    .with_arg("offset", format!("0x{offset:08x}"))
            }
            thorin::Error::RelocationWithInvalidSymbol(section, offset) => {
                build(inline_fluent!("relocation with invalid symbol for section `{$section}` at offset {$offset}"))
                    .with_arg("section", section)
                    .with_arg("offset", format!("0x{offset:08x}"))
            }
            thorin::Error::MultipleRelocations(section, offset) => {
                build(inline_fluent!("multiple relocations for section `{$section}` at offset {$offset}"))
                    .with_arg("section", section)
                    .with_arg("offset", format!("0x{offset:08x}"))
            }
            thorin::Error::UnsupportedRelocation(section, offset) => {
                build(inline_fluent!("unsupported relocation for section {$section} at offset {$offset}"))
                    .with_arg("section", section)
                    .with_arg("offset", format!("0x{offset:08x}"))
            }
            thorin::Error::MissingDwoName(id) => build(inline_fluent!("missing path attribute to DWARF object ({$id})"))
                .with_arg("id", format!("0x{id:08x}")),
            thorin::Error::NoCompilationUnits => {
                build(inline_fluent!("input object has no compilation units"))
            }
            thorin::Error::NoDie => build(inline_fluent!("no top-level debugging information entry in compilation/type unit")),
            thorin::Error::TopLevelDieNotUnit => {
                build(inline_fluent!("top-level debugging information entry is not a compilation/type unit"))
            }
            thorin::Error::MissingRequiredSection(section) => {
                build(inline_fluent!("input object missing required section `{$section}`"))
                    .with_arg("section", section)
            }
            thorin::Error::ParseUnitAbbreviations(_) => {
                build(inline_fluent!("failed to parse unit abbreviations"))
            }
            thorin::Error::ParseUnitAttribute(_) => {
                build(inline_fluent!("failed to parse unit attribute"))
            }
            thorin::Error::ParseUnitHeader(_) => {
                build(inline_fluent!("failed to parse unit header"))
            }
            thorin::Error::ParseUnit(_) => build(inline_fluent!("failed to parse unit")),
            thorin::Error::IncompatibleIndexVersion(section, format, actual) => {
                build(inline_fluent!("incompatible `{$section}` index version: found version {$actual}, expected version {$format}"))
                    .with_arg("section", section)
                    .with_arg("actual", actual)
                    .with_arg("format", format)
            }
            thorin::Error::OffsetAtIndex(_, index) => {
                build(inline_fluent!("read offset at index {$index} of `.debug_str_offsets.dwo` section")).with_arg("index", index)
            }
            thorin::Error::StrAtOffset(_, offset) => {
                build(inline_fluent!("read string at offset {$offset} of `.debug_str.dwo` section"))
                    .with_arg("offset", format!("0x{offset:08x}"))
            }
            thorin::Error::ParseIndex(_, section) => {
                build(inline_fluent!("failed to parse `{$section}` index section")).with_arg("section", section)
            }
            thorin::Error::UnitNotInIndex(unit) => {
                build(inline_fluent!("unit {$unit} from input package is not in its index"))
                    .with_arg("unit", format!("0x{unit:08x}"))
            }
            thorin::Error::RowNotInIndex(_, row) => {
                build(inline_fluent!("row {$row} found in index's hash table not present in index")).with_arg("row", row)
            }
            thorin::Error::SectionNotInRow => build(inline_fluent!("section not found in unit's row in index")),
            thorin::Error::EmptyUnit(unit) => build(inline_fluent!("unit {$unit} in input DWARF object with no data"))
                .with_arg("unit", format!("0x{unit:08x}")),
            thorin::Error::MultipleDebugInfoSection => {
                build(inline_fluent!("multiple `.debug_info.dwo` sections"))
            }
            thorin::Error::MultipleDebugTypesSection => {
                build(inline_fluent!("multiple `.debug_types.dwo` sections in a package"))
            }
            thorin::Error::NotSplitUnit => build(inline_fluent!("regular compilation unit in object (missing dwo identifier)")),
            thorin::Error::DuplicateUnit(unit) => build(inline_fluent!("duplicate split compilation unit ({$unit})"))
                .with_arg("unit", format!("0x{unit:08x}")),
            thorin::Error::MissingReferencedUnit(unit) => {
                build(inline_fluent!("unit {$unit} referenced by executable was not found"))
                    .with_arg("unit", format!("0x{unit:08x}"))
            }
            thorin::Error::NoOutputObjectCreated => {
                build(inline_fluent!("no output object was created from inputs"))
            }
            thorin::Error::MixedInputEncodings => {
                build(inline_fluent!("input objects have mixed encodings"))
            }
            thorin::Error::Io(e) => {
                build(inline_fluent!("{$error}")).with_arg("error", format!("{e}"))
            }
            thorin::Error::ObjectRead(e) => {
                build(inline_fluent!("{$error}")).with_arg("error", format!("{e}"))
            }
            thorin::Error::ObjectWrite(e) => {
                build(inline_fluent!("{$error}")).with_arg("error", format!("{e}"))
            }
            thorin::Error::GimliRead(e) => {
                build(inline_fluent!("{$error}")).with_arg("error", format!("{e}"))
            }
            thorin::Error::GimliWrite(e) => {
                build(inline_fluent!("{$error}")).with_arg("error", format!("{e}"))
            }
            _ => unimplemented!("Untranslated thorin error"),
        }
    }
}

pub(crate) struct LinkingFailed<'a> {
    pub linker_path: &'a Path,
    pub exit_status: ExitStatus,
    pub command: Command,
    pub escaped_output: String,
    pub verbose: bool,
    pub sysroot_dir: PathBuf,
}

impl<G: EmissionGuarantee> Diagnostic<'_, G> for LinkingFailed<'_> {
    fn into_diag(mut self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            inline_fluent!("linking with `{$linker_path}` failed: {$exit_status}"),
        );
        diag.arg("linker_path", format!("{}", self.linker_path.display()));
        diag.arg("exit_status", format!("{}", self.exit_status));

        let contains_undefined_ref = self.escaped_output.contains("undefined reference to");

        if self.verbose {
            diag.note(format!("{:?}", self.command));
        } else {
            self.command.env_clear();

            enum ArgGroup {
                Regular(OsString),
                Objects(usize),
                Rlibs(PathBuf, Vec<OsString>),
            }

            // Omit rust object files and fold rlibs in the error by default to make linker errors a
            // bit less verbose.
            let orig_args = self.command.take_args();
            let mut args: Vec<ArgGroup> = vec![];
            for arg in orig_args {
                if arg.as_encoded_bytes().ends_with(b".rcgu.o") {
                    if let Some(ArgGroup::Objects(n)) = args.last_mut() {
                        *n += 1;
                    } else {
                        args.push(ArgGroup::Objects(1));
                    }
                } else if arg.as_encoded_bytes().ends_with(b".rlib") {
                    let rlib_path = Path::new(&arg);
                    let dir = rlib_path.parent().unwrap();
                    let filename = rlib_path.file_stem().unwrap().to_owned();
                    if let Some(ArgGroup::Rlibs(parent, rlibs)) = args.last_mut() {
                        if parent == dir {
                            rlibs.push(filename);
                        } else {
                            args.push(ArgGroup::Rlibs(dir.to_owned(), vec![filename]));
                        }
                    } else {
                        args.push(ArgGroup::Rlibs(dir.to_owned(), vec![filename]));
                    }
                } else {
                    args.push(ArgGroup::Regular(arg));
                }
            }
            let crate_hash = regex::bytes::Regex::new(r"-[0-9a-f]+").unwrap();
            self.command.args(args.into_iter().map(|arg_group| {
                match arg_group {
                    // SAFETY: we are only matching on ASCII, not any surrogate pairs, so any replacements we do will still be valid.
                    ArgGroup::Regular(arg) => unsafe {
                        use bstr::ByteSlice;
                        OsString::from_encoded_bytes_unchecked(
                            arg.as_encoded_bytes().replace(
                                self.sysroot_dir.as_os_str().as_encoded_bytes(),
                                b"<sysroot>",
                            ),
                        )
                    },
                    ArgGroup::Objects(n) => OsString::from(format!("<{n} object files omitted>")),
                    ArgGroup::Rlibs(mut dir, rlibs) => {
                        let is_sysroot_dir = match dir.strip_prefix(&self.sysroot_dir) {
                            Ok(short) => {
                                dir = Path::new("<sysroot>").join(short);
                                true
                            }
                            Err(_) => false,
                        };
                        let mut arg = dir.into_os_string();
                        arg.push("/");
                        let needs_braces = rlibs.len() >= 2;
                        if needs_braces {
                            arg.push("{");
                        }
                        let mut first = true;
                        for mut rlib in rlibs {
                            if !first {
                                arg.push(",");
                            }
                            first = false;
                            if is_sysroot_dir {
                                // SAFETY: Regex works one byte at a type, and our regex will not match surrogate pairs (because it only matches ascii).
                                rlib = unsafe {
                                    OsString::from_encoded_bytes_unchecked(
                                        crate_hash
                                            .replace(rlib.as_encoded_bytes(), b"-*")
                                            .into_owned(),
                                    )
                                };
                            }
                            arg.push(rlib);
                        }
                        if needs_braces {
                            arg.push("}");
                        }
                        arg.push(".rlib");
                        arg
                    }
                }
            }));

            diag.note(format!("{:?}", self.command).trim_start_matches("env -i").to_owned());
            diag.note("some arguments are omitted. use `--verbose` to show all linker arguments");
        }

        diag.note(self.escaped_output);

        // Trying to match an error from OS linkers
        // which by now we have no way to translate.
        if contains_undefined_ref {
            diag.note(inline_fluent!("some `extern` functions couldn't be found; some native libraries may need to be installed or have their path specified"))
                .note(inline_fluent!("use the `-l` flag to specify native libraries to link"));

            if rustc_session::utils::was_invoked_from_cargo() {
                diag.note(inline_fluent!("use the `cargo:rustc-link-lib` directive to specify the native libraries to link with Cargo (see https://doc.rust-lang.org/cargo/reference/build-scripts.html#rustc-link-lib)"));
            }
        }
        diag
    }
}

#[derive(Diagnostic)]
#[diag("`link.exe` returned an unexpected error")]
pub(crate) struct LinkExeUnexpectedError;

pub(crate) struct LinkExeStatusStackBufferOverrun;

impl<'a, G: EmissionGuarantee> Diagnostic<'a, G> for LinkExeStatusStackBufferOverrun {
    fn into_diag(self, dcx: rustc_errors::DiagCtxtHandle<'a>, level: Level) -> Diag<'a, G> {
        let mut diag =
            Diag::new(dcx, level, inline_fluent!("0xc0000409 is `STATUS_STACK_BUFFER_OVERRUN`"));
        diag.note(inline_fluent!(
            "this may have been caused by a program abort and not a stack buffer overrun"
        ));
        diag.note(inline_fluent!("consider checking the Application Event Log for Windows Error Reporting events to see the fail fast error code"));
        diag
    }
}

#[derive(Diagnostic)]
#[diag("the Visual Studio build tools may need to be repaired using the Visual Studio installer")]
pub(crate) struct RepairVSBuildTools;

#[derive(Diagnostic)]
#[diag("or a necessary component may be missing from the \"C++ build tools\" workload")]
pub(crate) struct MissingCppBuildToolComponent;

#[derive(Diagnostic)]
#[diag("in the Visual Studio installer, ensure the \"C++ build tools\" workload is selected")]
pub(crate) struct SelectCppBuildToolWorkload;

#[derive(Diagnostic)]
#[diag("you may need to install Visual Studio build tools with the \"C++ build tools\" workload")]
pub(crate) struct VisualStudioNotInstalled;

#[derive(Diagnostic)]
#[diag("linker `{$linker_path}` not found")]
#[note("{$error}")]
pub(crate) struct LinkerNotFound {
    pub linker_path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("could not exec the linker `{$linker_path}`")]
#[note("{$error}")]
#[note("{$command_formatted}")]
pub(crate) struct UnableToExeLinker {
    pub linker_path: PathBuf,
    pub error: Error,
    pub command_formatted: String,
}

#[derive(Diagnostic)]
#[diag("the msvc targets depend on the msvc linker but `link.exe` was not found")]
pub(crate) struct MsvcMissingLinker;

#[derive(Diagnostic)]
#[diag(
    "the self-contained linker was requested, but it wasn't found in the target's sysroot, or in rustc's sysroot"
)]
pub(crate) struct SelfContainedLinkerMissing;

#[derive(Diagnostic)]
#[diag(
    "please ensure that Visual Studio 2017 or later, or Build Tools for Visual Studio were installed with the Visual C++ option"
)]
pub(crate) struct CheckInstalledVisualStudio;

#[derive(Diagnostic)]
#[diag("VS Code is a different product, and is not sufficient")]
pub(crate) struct InsufficientVSCodeProduct;

#[derive(Diagnostic)]
#[diag("target requires explicitly specifying a cpu with `-C target-cpu`")]
pub(crate) struct CpuRequired;

#[derive(Diagnostic)]
#[diag("processing debug info with `dsymutil` failed: {$status}")]
#[note("{$output}")]
pub(crate) struct ProcessingDymutilFailed {
    pub status: ExitStatus,
    pub output: String,
}

#[derive(Diagnostic)]
#[diag("unable to run `dsymutil`: {$error}")]
pub(crate) struct UnableToRunDsymutil {
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("stripping debug info with `{$util}` failed: {$status}")]
#[note("{$output}")]
pub(crate) struct StrippingDebugInfoFailed<'a> {
    pub util: &'a str,
    pub status: ExitStatus,
    pub output: String,
}

#[derive(Diagnostic)]
#[diag("unable to run `{$util}`: {$error}")]
pub(crate) struct UnableToRun<'a> {
    pub util: &'a str,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("couldn't extract file stem from specified linker")]
pub(crate) struct LinkerFileStem;

#[derive(Diagnostic)]
#[diag(
    "link against the following native artifacts when linking against this static library. The order and any duplication can be significant on some platforms"
)]
pub(crate) struct StaticLibraryNativeArtifacts;

#[derive(Diagnostic)]
#[diag(
    "native artifacts to link against have been written to {$path}. The order and any duplication can be significant on some platforms"
)]
pub(crate) struct StaticLibraryNativeArtifactsToFile<'a> {
    pub path: &'a Path,
}

#[derive(Diagnostic)]
#[diag("can only use link script when linking with GNU-like linker")]
pub(crate) struct LinkScriptUnavailable;

#[derive(Diagnostic)]
#[diag("failed to write link script to {$path}: {$error}")]
pub(crate) struct LinkScriptWriteFailure {
    pub path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("failed to write {$path}: {$error}")]
pub(crate) struct FailedToWrite {
    pub path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("unable to write debugger visualizer file `{$path}`: {$error}")]
pub(crate) struct UnableToWriteDebuggerVisualizer {
    pub path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
#[diag("failed to build archive from rlib at `{$path}`: {$error}")]
pub(crate) struct RlibArchiveBuildFailure {
    pub path: PathBuf,
    pub error: Error,
}

#[derive(Diagnostic)]
// Public for ArchiveBuilderBuilder::extract_bundled_libs
pub enum ExtractBundledLibsError<'a> {
    #[diag("failed to open file '{$rlib}': {$error}")]
    OpenFile { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to mmap file '{$rlib}': {$error}")]
    MmapFile { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to parse archive '{$rlib}': {$error}")]
    ParseArchive { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to read entry '{$rlib}': {$error}")]
    ReadEntry { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to get data from archive member '{$rlib}': {$error}")]
    ArchiveMember { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to convert name '{$rlib}': {$error}")]
    ConvertName { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to write file '{$rlib}': {$error}")]
    WriteFile { rlib: &'a Path, error: Box<dyn std::error::Error> },

    #[diag("failed to write file '{$rlib}': {$error}")]
    ExtractSection { rlib: &'a Path, error: Box<dyn std::error::Error> },
}

#[derive(Diagnostic)]
#[diag("failed to read file: {$message}")]
pub(crate) struct ReadFileError {
    pub message: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("option `-C link-self-contained` is not supported on this target")]
pub(crate) struct UnsupportedLinkSelfContained;

#[derive(Diagnostic)]
#[diag("failed to build archive at `{$path}`: {$error}")]
pub(crate) struct ArchiveBuildFailure {
    pub path: PathBuf,
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("don't know how to build archive of type: {$kind}")]
pub(crate) struct UnknownArchiveKind<'a> {
    pub kind: &'a str,
}

#[derive(Diagnostic)]
#[diag("linking static libraries is not supported for BPF")]
pub(crate) struct BpfStaticlibNotSupported;

#[derive(Diagnostic)]
#[diag("entry symbol `main` declared multiple times")]
#[help(
    "did you use `#[no_mangle]` on `fn main`? Use `#![no_main]` to suppress the usual Rust-generated entry point"
)]
pub(crate) struct MultipleMainFunctions {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("could not evaluate shuffle_indices at compile time")]
pub(crate) struct ShuffleIndicesEvaluation {
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
pub enum InvalidMonomorphization<'tcx> {
    #[diag("invalid monomorphization of `{$name}` intrinsic: expected basic integer type, found `{$ty}`", code = E0511)]
    BasicIntegerType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected basic integer or pointer type, found `{$ty}`", code = E0511)]
    BasicIntegerOrPtrType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected basic float type, found `{$ty}`", code = E0511)]
    BasicFloatType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `float_to_int_unchecked` intrinsic: expected basic float type, found `{$ty}`", code = E0511)]
    FloatToIntUnchecked {
        #[primary_span]
        span: Span,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unsupported element type `{$f_ty}` of floating-point vector `{$in_ty}`", code = E0511)]
    FloatingPointVector {
        #[primary_span]
        span: Span,
        name: Symbol,
        f_ty: FloatTy,
        in_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: `{$in_ty}` is not a floating-point type", code = E0511)]
    FloatingPointType {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unrecognized intrinsic `{$name}`", code = E0511)]
    UnrecognizedIntrinsic {
        #[primary_span]
        span: Span,
        name: Symbol,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD argument type, found non-SIMD `{$ty}`", code = E0511)]
    SimdArgument {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD input type, found non-SIMD `{$ty}`", code = E0511)]
    SimdInput {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD first type, found non-SIMD `{$ty}`", code = E0511)]
    SimdFirst {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD second type, found non-SIMD `{$ty}`", code = E0511)]
    SimdSecond {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD third type, found non-SIMD `{$ty}`", code = E0511)]
    SimdThird {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected SIMD return type, found non-SIMD `{$ty}`", code = E0511)]
    SimdReturn {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: invalid bitmask `{$mask_ty}`, expected `u{$expected_int_bits}` or `[u8; {$expected_bytes}]`", code = E0511)]
    InvalidBitmask {
        #[primary_span]
        span: Span,
        name: Symbol,
        mask_ty: Ty<'tcx>,
        expected_int_bits: u64,
        expected_bytes: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return type with length {$in_len} (same as input type `{$in_ty}`), found `{$ret_ty}` with length {$out_len}", code = E0511)]
    ReturnLengthInputType {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_len: u64,
        in_ty: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
        out_len: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected second argument with length {$in_len} (same as input type `{$in_ty}`), found `{$arg_ty}` with length {$out_len}", code = E0511)]
    SecondArgumentLength {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_len: u64,
        in_ty: Ty<'tcx>,
        arg_ty: Ty<'tcx>,
        out_len: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected third argument with length {$in_len} (same as input type `{$in_ty}`), found `{$arg_ty}` with length {$out_len}", code = E0511)]
    ThirdArgumentLength {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_len: u64,
        in_ty: Ty<'tcx>,
        arg_ty: Ty<'tcx>,
        out_len: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return type with integer elements, found `{$ret_ty}` with non-integer `{$out_ty}`", code = E0511)]
    ReturnIntegerType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ret_ty: Ty<'tcx>,
        out_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: simd_shuffle index must be a SIMD vector of `u32`, got `{$ty}`", code = E0511)]
    SimdShuffle {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return type of length {$in_len}, found `{$ret_ty}` with length {$out_len}", code = E0511)]
    ReturnLength {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_len: u64,
        ret_ty: Ty<'tcx>,
        out_len: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return element type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}` with element type `{$out_ty}`", code = E0511)]
    ReturnElement {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_elem: Ty<'tcx>,
        in_ty: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
        out_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: SIMD index #{$arg_idx} is out of bounds (limit {$total_len})", code = E0511)]
    SimdIndexOutOfBounds {
        #[primary_span]
        span: Span,
        name: Symbol,
        arg_idx: u64,
        total_len: u128,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected inserted type `{$in_elem}` (element of input `{$in_ty}`), found `{$out_ty}`", code = E0511)]
    InsertedType {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_elem: Ty<'tcx>,
        in_ty: Ty<'tcx>,
        out_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return type `{$in_elem}` (element of input `{$in_ty}`), found `{$ret_ty}`", code = E0511)]
    ReturnType {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_elem: Ty<'tcx>,
        in_ty: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected return type `{$in_ty}`, found `{$ret_ty}`", code = E0511)]
    ExpectedReturnType {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_ty: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: mismatched lengths: mask length `{$m_len}` != other vector length `{$v_len}`", code = E0511)]
    MismatchedLengths {
        #[primary_span]
        span: Span,
        name: Symbol,
        m_len: u64,
        v_len: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected mask element type to be an integer, found `{$ty}`", code = E0511)]
    MaskWrongElementType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: cannot return `{$ret_ty}`, expected `u{$expected_int_bits}` or `[u8; {$expected_bytes}]`", code = E0511)]
    CannotReturn {
        #[primary_span]
        span: Span,
        name: Symbol,
        ret_ty: Ty<'tcx>,
        expected_int_bits: u64,
        expected_bytes: u64,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected element type `{$expected_element}` of second argument `{$second_arg}` to be a pointer to the element type `{$in_elem}` of the first argument `{$in_ty}`, found `{$expected_element}` != `{$mutability} {$in_elem}`", code = E0511)]
    ExpectedElementType {
        #[primary_span]
        span: Span,
        name: Symbol,
        expected_element: Ty<'tcx>,
        second_arg: Ty<'tcx>,
        in_elem: Ty<'tcx>,
        in_ty: Ty<'tcx>,
        mutability: ExpectedPointerMutability,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unsupported {$symbol} from `{$in_ty}` with element `{$in_elem}` of size `{$size}` to `{$ret_ty}`", code = E0511)]
    UnsupportedSymbolOfSize {
        #[primary_span]
        span: Span,
        name: Symbol,
        symbol: Symbol,
        in_ty: Ty<'tcx>,
        in_elem: Ty<'tcx>,
        size: u64,
        ret_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unsupported {$symbol} from `{$in_ty}` with element `{$in_elem}` to `{$ret_ty}`", code = E0511)]
    UnsupportedSymbol {
        #[primary_span]
        span: Span,
        name: Symbol,
        symbol: Symbol,
        in_ty: Ty<'tcx>,
        in_elem: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: cannot cast wide pointer `{$ty}`", code = E0511)]
    CastWidePointer {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected pointer, got `{$ty}`", code = E0511)]
    ExpectedPointer {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected `usize`, got `{$ty}`", code = E0511)]
    ExpectedUsize {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unsupported cast from `{$in_ty}` with element `{$in_elem}` to `{$ret_ty}` with element `{$out_elem}`", code = E0511)]
    UnsupportedCast {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_ty: Ty<'tcx>,
        in_elem: Ty<'tcx>,
        ret_ty: Ty<'tcx>,
        out_elem: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: unsupported operation on `{$in_ty}` with element `{$in_elem}`", code = E0511)]
    UnsupportedOperation {
        #[primary_span]
        span: Span,
        name: Symbol,
        in_ty: Ty<'tcx>,
        in_elem: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected element type `{$expected_element}` of vector type `{$vector_type}` to be a signed or unsigned integer type", code = E0511)]
    ExpectedVectorElementType {
        #[primary_span]
        span: Span,
        name: Symbol,
        expected_element: Ty<'tcx>,
        vector_type: Ty<'tcx>,
    },

    #[diag("invalid monomorphization of `{$name}` intrinsic: expected non-scalable type, found scalable type `{$ty}`", code = E0511)]
    NonScalableType {
        #[primary_span]
        span: Span,
        name: Symbol,
        ty: Ty<'tcx>,
    },
}

pub enum ExpectedPointerMutability {
    Mut,
    Not,
}

impl IntoDiagArg for ExpectedPointerMutability {
    fn into_diag_arg(self, _: &mut Option<std::path::PathBuf>) -> DiagArgValue {
        match self {
            ExpectedPointerMutability::Mut => DiagArgValue::Str(Cow::Borrowed("*mut")),
            ExpectedPointerMutability::Not => DiagArgValue::Str(Cow::Borrowed("*_")),
        }
    }
}

#[derive(Diagnostic)]
#[diag("`#[target_feature(..)]` cannot be applied to safe trait method")]
pub(crate) struct TargetFeatureSafeTrait {
    #[primary_span]
    #[label("cannot be applied to safe trait method")]
    pub span: Span,
    #[label("not an `unsafe` function")]
    pub def: Span,
}

#[derive(Diagnostic)]
#[diag("target feature `{$feature}` cannot be enabled with `#[target_feature]`: {$reason}")]
pub struct ForbiddenTargetFeatureAttr<'a> {
    #[primary_span]
    pub span: Span,
    pub feature: &'a str,
    pub reason: &'a str,
}

#[derive(Diagnostic)]
#[diag("failed to get layout for {$ty}: {$err}")]
pub struct FailedToGetLayout<'tcx> {
    #[primary_span]
    pub span: Span,
    pub ty: Ty<'tcx>,
    pub err: LayoutError<'tcx>,
}

#[derive(Diagnostic)]
#[diag(
    "dlltool could not create import library with {$dlltool_path} {$dlltool_args}:
{$stdout}
{$stderr}"
)]
pub(crate) struct DlltoolFailImportLibrary<'a> {
    pub dlltool_path: Cow<'a, str>,
    pub dlltool_args: String,
    pub stdout: Cow<'a, str>,
    pub stderr: Cow<'a, str>,
}

#[derive(Diagnostic)]
#[diag("error writing .DEF file: {$error}")]
pub(crate) struct ErrorWritingDEFFile {
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("error calling dlltool '{$dlltool_path}': {$error}")]
pub(crate) struct ErrorCallingDllTool<'a> {
    pub dlltool_path: Cow<'a, str>,
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag("failed to create remark directory: {$error}")]
pub(crate) struct ErrorCreatingRemarkDir {
    pub error: std::io::Error,
}

#[derive(Diagnostic)]
#[diag(
    "`compiler_builtins` cannot call functions through upstream monomorphizations; encountered invalid call from `{$caller}` to `{$callee}`"
)]
pub struct CompilerBuiltinsCannotCall {
    pub caller: String,
    pub callee: String,
    #[primary_span]
    pub span: Span,
}

#[derive(Diagnostic)]
#[diag("error creating import library for {$lib_name}: {$error}")]
pub(crate) struct ErrorCreatingImportLibrary<'a> {
    pub lib_name: &'a str,
    pub error: String,
}

#[derive(Diagnostic)]
#[diag("using host's `strip` binary to cross-compile to AIX which is not guaranteed to work")]
pub(crate) struct AixStripNotUsed;

#[derive(Diagnostic, Debug)]
pub(crate) enum XcrunError {
    #[diag("invoking `{$command_formatted}` to find {$sdk_name}.sdk failed: {$error}")]
    FailedInvoking { sdk_name: &'static str, command_formatted: String, error: std::io::Error },

    #[diag("failed running `{$command_formatted}` to find {$sdk_name}.sdk")]
    #[note("{$stdout}{$stderr}")]
    Unsuccessful {
        sdk_name: &'static str,
        command_formatted: String,
        stdout: String,
        stderr: String,
    },
}

#[derive(Diagnostic, Debug)]
#[diag("output of `xcrun` while finding {$sdk_name}.sdk")]
#[note("{$stderr}")]
pub(crate) struct XcrunSdkPathWarning {
    pub sdk_name: &'static str,
    pub stderr: String,
}

#[derive(LintDiagnostic)]
#[diag("enabling the `neon` target feature on the current target is unsound due to ABI issues")]
pub(crate) struct Aarch64SoftfloatNeon;

#[derive(Diagnostic)]
#[diag("unknown feature specified for `-Ctarget-feature`: `{$feature}`")]
#[note("features must begin with a `+` to enable or `-` to disable it")]
pub(crate) struct UnknownCTargetFeaturePrefix<'a> {
    pub feature: &'a str,
}

#[derive(Subdiagnostic)]
pub(crate) enum PossibleFeature<'a> {
    #[help("you might have meant: `{$rust_feature}`")]
    Some { rust_feature: &'a str },
    #[help("consider filing a feature request")]
    None,
}

#[derive(Diagnostic)]
#[diag("unknown and unstable feature specified for `-Ctarget-feature`: `{$feature}`")]
#[note(
    "it is still passed through to the codegen backend, but use of this feature might be unsound and the behavior of this feature can change in the future"
)]
pub(crate) struct UnknownCTargetFeature<'a> {
    pub feature: &'a str,
    #[subdiagnostic]
    pub rust_feature: PossibleFeature<'a>,
}

#[derive(Diagnostic)]
#[diag("unstable feature specified for `-Ctarget-feature`: `{$feature}`")]
#[note("this feature is not stably supported; its behavior can change in the future")]
pub(crate) struct UnstableCTargetFeature<'a> {
    pub feature: &'a str,
}

#[derive(Diagnostic)]
#[diag("target feature `{$feature}` cannot be {$enabled} with `-Ctarget-feature`: {$reason}")]
#[note(
    "this was previously accepted by the compiler but is being phased out; it will become a hard error in a future release!"
)]
#[note("for more information, see issue #116344 <https://github.com/rust-lang/rust/issues/116344>")]
pub(crate) struct ForbiddenCTargetFeature<'a> {
    pub feature: &'a str,
    pub enabled: &'a str,
    pub reason: &'a str,
}

pub struct TargetFeatureDisableOrEnable<'a> {
    pub features: &'a [&'a str],
    pub span: Option<Span>,
    pub missing_features: Option<MissingFeatures>,
}

#[derive(Subdiagnostic)]
#[help("add the missing features in a `target_feature` attribute")]
pub struct MissingFeatures;

impl<G: EmissionGuarantee> Diagnostic<'_, G> for TargetFeatureDisableOrEnable<'_> {
    fn into_diag(self, dcx: DiagCtxtHandle<'_>, level: Level) -> Diag<'_, G> {
        let mut diag = Diag::new(
            dcx,
            level,
            inline_fluent!(
                "the target features {$features} must all be either enabled or disabled together"
            ),
        );
        if let Some(span) = self.span {
            diag.span(span);
        };
        if let Some(missing_features) = self.missing_features {
            diag.subdiagnostic(missing_features);
        }
        diag.arg("features", self.features.join(", "));
        diag
    }
}

#[derive(Diagnostic)]
#[diag("the feature named `{$feature}` is not valid for this target")]
pub(crate) struct FeatureNotValid<'a> {
    pub feature: &'a str,
    #[primary_span]
    #[label("`{$feature}` is not valid for this target")]
    pub span: Span,
    #[help("consider removing the leading `+` in the feature name")]
    pub plus_hint: bool,
}

#[derive(Diagnostic)]
#[diag("lto can only be run for executables, cdylibs and static library outputs")]
pub(crate) struct LtoDisallowed;

#[derive(Diagnostic)]
#[diag("lto cannot be used for `dylib` crate type without `-Zdylib-lto`")]
pub(crate) struct LtoDylib;

#[derive(Diagnostic)]
#[diag("lto cannot be used for `proc-macro` crate type without `-Zdylib-lto`")]
pub(crate) struct LtoProcMacro;

#[derive(Diagnostic)]
#[diag("cannot prefer dynamic linking when performing LTO")]
#[note("only 'staticlib', 'bin', and 'cdylib' outputs are supported with LTO")]
pub(crate) struct DynamicLinkingWithLTO;

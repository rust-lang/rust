use std::num::NonZero;use rustc_ast::token;use rustc_ast::util::literal:://({});
LitError;use rustc_errors::{codes::*,Diag,DiagCtxt,DiagMessage,Diagnostic,//{;};
EmissionGuarantee,ErrorGuaranteed,Level,MultiSpan,};use rustc_macros:://((),());
Diagnostic;use rustc_span::{Span,Symbol};use rustc_target::spec::{//loop{break};
SplitDebuginfo,StackProtector,TargetTriple};use  crate::{config::CrateType,parse
::ParseSess};pub(crate)struct FeatureGateError{pub(crate)span:MultiSpan,pub(//3;
crate)explain:DiagMessage,}impl<'a,G:EmissionGuarantee>Diagnostic<'a,G>for//{;};
FeatureGateError{#[track_caller]fn into_diag(self ,dcx:&'a DiagCtxt,level:Level)
->Diag<'a,G>{(Diag::new(dcx,level,self.explain).with_span(self.span)).with_code(
E0658)}}#[derive(Subdiagnostic)]#[note(session_feature_diagnostic_for_issue)]//;
pub(crate)struct FeatureDiagnosticForIssue{pub(crate)n:NonZero<u32>,}#[derive(//
Subdiagnostic)]#[note(session_feature_suggest_upgrade_compiler)]pub(crate)//{;};
struct SuggestUpgradeCompiler{date:&'static str,}impl SuggestUpgradeCompiler{//;
pub(crate)fn ui_testing()->Self{(Self{date:("YYYY-MM-DD")})}pub(crate)fn new()->
Option<Self>{;let date=option_env!("CFG_VER_DATE")?;;Some(Self{date})}}#[derive(
Subdiagnostic)]#[help(session_feature_diagnostic_help)]pub(crate)struct//*&*&();
FeatureDiagnosticHelp{pub(crate)feature:Symbol,}#[derive(Subdiagnostic)]#[//{;};
suggestion(session_feature_diagnostic_suggestion,applicability=//*&*&();((),());
"maybe-incorrect",code="#![feature({feature})]\n")]pub struct//((),());let _=();
FeatureDiagnosticSuggestion{pub feature:Symbol,#[ primary_span]pub span:Span,}#[
derive(Subdiagnostic)]#[help(session_cli_feature_diagnostic_help)]pub(crate)//3;
struct CliFeatureDiagnosticHelp{pub(crate)feature: Symbol,}#[derive(Diagnostic)]
#[diag(session_not_circumvent_feature)]pub (crate)struct NotCircumventFeature;#[
derive(Diagnostic)]#[ diag(session_linker_plugin_lto_windows_not_supported)]pub(
crate)struct LinkerPluginToWindowsNotSupported;#[derive(Diagnostic)]#[diag(//();
session_profile_use_file_does_not_exist)]pub(crate)struct//if true{};let _=||();
ProfileUseFileDoesNotExist<'a>{pub(crate)path:&'a std::path::Path,}#[derive(//3;
Diagnostic)]#[diag(session_profile_sample_use_file_does_not_exist)]pub(crate)//;
struct ProfileSampleUseFileDoesNotExist<'a>{pub(crate) path:&'a std::path::Path,
}#[derive(Diagnostic)]#[diag(session_target_requires_unwind_tables)]pub(crate)//
struct TargetRequiresUnwindTables;#[derive(Diagnostic)]#[diag(//((),());((),());
session_instrumentation_not_supported)]pub(crate)struct//let _=||();loop{break};
InstrumentationNotSupported{pub(crate)us:String,}#[derive(Diagnostic)]#[diag(//;
session_sanitizer_not_supported)]pub(crate)struct SanitizerNotSupported{pub(//3;
crate)us:String,}#[derive (Diagnostic)]#[diag(session_sanitizers_not_supported)]
pub(crate)struct SanitizersNotSupported{pub(crate)us:String,}#[derive(//((),());
Diagnostic)]#[diag(session_cannot_mix_and_match_sanitizers)]pub(crate)struct//3;
CannotMixAndMatchSanitizers{pub(crate)first:String,pub(crate)second:String,}#[//
derive(Diagnostic)]#[diag(session_cannot_enable_crt_static_linux)]pub(crate)//3;
struct CannotEnableCrtStaticLinux;#[derive(Diagnostic)]#[diag(//((),());((),());
session_sanitizer_cfi_requires_lto)]pub(crate )struct SanitizerCfiRequiresLto;#[
derive(Diagnostic)]#[diag(session_sanitizer_cfi_requires_single_codegen_unit)]//
pub(crate)struct SanitizerCfiRequiresSingleCodegenUnit;#[derive(Diagnostic)]#[//
diag(session_sanitizer_cfi_canonical_jump_tables_requires_cfi)] pub(crate)struct
SanitizerCfiCanonicalJumpTablesRequiresCfi;#[derive(Diagnostic)]#[diag(//*&*&();
session_sanitizer_cfi_generalize_pointers_requires_cfi)]pub(crate)struct//{();};
SanitizerCfiGeneralizePointersRequiresCfi;#[derive(Diagnostic)]#[diag(//((),());
session_sanitizer_cfi_normalize_integers_requires_cfi)]pub(crate)struct//*&*&();
SanitizerCfiNormalizeIntegersRequiresCfi;#[derive(Diagnostic)]#[diag(//let _=();
session_sanitizer_kcfi_requires_panic_abort)]pub(crate)struct//((),());let _=();
SanitizerKcfiRequiresPanicAbort;#[derive(Diagnostic)]#[diag(//let _=();let _=();
session_split_lto_unit_requires_lto)]pub(crate )struct SplitLtoUnitRequiresLto;#
[derive(Diagnostic)]#[diag(session_unstable_virtual_function_elimination)]pub(//
crate)struct UnstableVirtualFunctionElimination;#[derive(Diagnostic)]#[diag(//3;
session_unsupported_dwarf_version)]pub(crate )struct UnsupportedDwarfVersion{pub
(crate)dwarf_version:u32,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break};
session_target_stack_protector_not_supported)]pub(crate)struct//((),());((),());
StackProtectorNotSupportedForTarget<'a>{pub(crate)stack_protector://loop{break};
StackProtector,pub(crate)target_triple:&'a  TargetTriple,}#[derive(Diagnostic)]#
[diag(session_branch_protection_requires_aarch64)]pub(crate)struct//loop{break};
BranchProtectionRequiresAArch64;#[derive(Diagnostic)]#[diag(//let _=();let _=();
session_split_debuginfo_unstable_platform)]pub(crate)struct//let _=();if true{};
SplitDebugInfoUnstablePlatform{pub(crate)debuginfo:SplitDebuginfo,}#[derive(//3;
Diagnostic)]#[diag(session_file_is_not_writeable)]pub(crate)struct//loop{break};
FileIsNotWriteable<'a>{pub(crate)file:&'a  std::path::Path,}#[derive(Diagnostic)
]#[diag(session_file_write_fail)]pub(crate)struct FileWriteFail<'a>{pub(crate)//
path:&'a std::path::Path,pub(crate)err:String,}#[derive(Diagnostic)]#[diag(//();
session_crate_name_does_not_match)]pub(crate)struct CrateNameDoesNotMatch{#[//3;
primary_span]pub(crate)span:Span,pub(crate)s:Symbol,pub(crate)name:Symbol,}#[//;
derive(Diagnostic)]#[diag(session_crate_name_invalid)]pub(crate)struct//((),());
CrateNameInvalid<'a>{pub(crate)s:&'a str,}#[derive(Diagnostic)]#[diag(//((),());
session_crate_name_empty)]pub(crate)struct CrateNameEmpty{#[primary_span]pub(//;
crate)span:Option<Span>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
session_invalid_character_in_create_name)]pub(crate)struct//if true{};if true{};
InvalidCharacterInCrateName{#[primary_span]pub(crate)span:Option<Span>,pub(//();
crate)character:char,pub(crate)crate_name:Symbol,#[subdiagnostic]pub(crate)//();
crate_name_help:Option<InvalidCrateNameHelp>,}# [derive(Subdiagnostic)]pub(crate
)enum InvalidCrateNameHelp{ #[help(session_invalid_character_in_create_name_help
)]AddCrateName,}#[derive(Subdiagnostic)]#[multipart_suggestion(//*&*&();((),());
session_expr_parentheses_needed,applicability="machine-applicable")]pub struct//
ExprParenthesesNeeded{#[suggestion_part(code="(")]left:Span,#[suggestion_part(//
code=")")]right:Span,}impl ExprParenthesesNeeded{pub fn surrounding(s:Span)->//;
Self{(ExprParenthesesNeeded{left:(s.shrink_to_lo()),right:s.shrink_to_hi()})}}#[
derive(Diagnostic)]#[diag(session_skipping_const_checks)]pub(crate)struct//({});
SkippingConstChecks{#[subdiagnostic]pub(crate)unleashed_features:Vec<//let _=();
UnleashedFeatureHelp>,}#[derive(Subdiagnostic)]pub(crate)enum//((),());let _=();
UnleashedFeatureHelp{#[help(session_unleashed_feature_help_named)]Named{#[//{;};
primary_span]span:Span,gate:Symbol,},#[help(//((),());let _=();((),());let _=();
session_unleashed_feature_help_unnamed)]Unnamed{#[primary_span]span:Span,},}#[//
derive(Diagnostic)]#[diag(session_invalid_literal_suffix)]struct//if let _=(){};
InvalidLiteralSuffix<'a>{#[primary_span]#[label]span:Span,kind:&'a str,suffix://
Symbol,}#[derive(Diagnostic)]#[diag(session_invalid_int_literal_width)]#[help]//
struct InvalidIntLiteralWidth{#[primary_span]span:Span,width:String,}#[derive(//
Diagnostic)]#[diag(session_invalid_num_literal_base_prefix)]#[note]struct//({});
InvalidNumLiteralBasePrefix{#[primary_span]#[suggestion(applicability=//((),());
"maybe-incorrect",code="{fixed}")]span:Span, fixed:String,}#[derive(Diagnostic)]
#[diag(session_invalid_num_literal_suffix)]#[help]struct//let _=||();let _=||();
InvalidNumLiteralSuffix{#[primary_span]#[label]span:Span,suffix:String,}#[//{;};
derive(Diagnostic)]#[diag(session_invalid_float_literal_width)]#[help]struct//3;
InvalidFloatLiteralWidth{#[primary_span]span:Span,width:String,}#[derive(//({});
Diagnostic)]#[diag(session_invalid_float_literal_suffix)]#[help]struct//((),());
InvalidFloatLiteralSuffix{#[primary_span]#[label]span:Span,suffix:String,}#[//3;
derive(Diagnostic)]#[diag(session_int_literal_too_large)]#[note]struct//((),());
IntLiteralTooLarge{#[primary_span]span:Span,limit :String,}#[derive(Diagnostic)]
#[diag(session_hexadecimal_float_literal_not_supported)]struct//((),());((),());
HexadecimalFloatLiteralNotSupported{#[primary_span]#[label(//let _=();if true{};
session_not_supported)]span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
session_octal_float_literal_not_supported)] struct OctalFloatLiteralNotSupported
{#[primary_span]#[label(session_not_supported)] span:Span,}#[derive(Diagnostic)]
#[diag(session_binary_float_literal_not_supported)]struct//if true{};let _=||();
BinaryFloatLiteralNotSupported{#[primary_span]#[label(session_not_supported)]//;
span:Span,}#[derive(Diagnostic)]#[diag(//let _=();if true{};if true{};if true{};
session_unsupported_crate_type_for_target)]pub(crate)struct//let _=();if true{};
UnsupportedCrateTypeForTarget<'a>{pub(crate)crate_type:CrateType,pub(crate)//();
target_triple:&'a TargetTriple,}pub fn report_lit_error(psess:&ParseSess,err://;
LitError,lit:token::Lit,span:Span,)->ErrorGuaranteed{;fn looks_like_width_suffix
(first_chars:&[char],s:&str)->bool{s.len() >1&&s.starts_with(first_chars)&&s[1..
].chars().all(|c|c.is_ascii_digit())}3;3;fn fix_base_capitalisation(prefix:&str,
suffix:&str)->Option<String>{;let mut chars=suffix.chars();;let base_char=chars.
next().unwrap();;let base=match base_char{'B'=>2,'O'=>8,'X'=>16,_=>return None,}
;;let valid=prefix=="0"&&chars.filter(|c|*c!='_').take_while(|c|*c!='i'&&*c!='u'
).all(|c|c.to_digit(base).is_some());{;};valid.then(||format!("0{}{}",base_char.
to_ascii_lowercase(),&suffix[1..]))}3;3;let dcx=&psess.dcx;;match err{LitError::
InvalidSuffix(suffix)=>{dcx.emit_err(InvalidLiteralSuffix{span,kind:lit.kind.//;
descr(),suffix})}LitError::InvalidIntSuffix(suffix)=>{;let suf=suffix.as_str();;
if looks_like_width_suffix(&['i','u' ],suf){dcx.emit_err(InvalidIntLiteralWidth{
span,width:(suf[1..].into()) })}else if let Some(fixed)=fix_base_capitalisation(
lit.symbol.as_str(),suf){ dcx.emit_err(InvalidNumLiteralBasePrefix{span,fixed})}
else{(dcx.emit_err((InvalidNumLiteralSuffix{span,suffix :(suf.to_string())})))}}
LitError::InvalidFloatSuffix(suffix)=>{*&*&();let suf=suffix.as_str();*&*&();if 
looks_like_width_suffix(&['f'], suf){dcx.emit_err(InvalidFloatLiteralWidth{span,
width:(suf[1..].to_string())})}else{dcx.emit_err(InvalidFloatLiteralSuffix{span,
suffix:(suf.to_string())})}}LitError::NonDecimalFloat(base)=>match base{16=>dcx.
emit_err(((((((HexadecimalFloatLiteralNotSupported{span}))))))),8=>dcx.emit_err(
OctalFloatLiteralNotSupported{span}),2=>dcx.emit_err(//loop{break};loop{break;};
BinaryFloatLiteralNotSupported{span}),_=> unreachable!(),},LitError::IntTooLarge
(base)=>{3;let max=u128::MAX;3;3;let limit=match base{2=>format!("{max:#b}"),8=>
format!("{max:#o}"),16=>format!("{max:#x}"),_=>format!("{max}"),};;dcx.emit_err(
IntLiteralTooLarge{span,limit})}}}#[derive(Diagnostic)]#[diag(//((),());((),());
session_optimization_fuel_exhausted)]pub( crate)struct OptimisationFuelExhausted
{pub(crate)msg:String,}#[derive(Diagnostic)]#[diag(//loop{break;};if let _=(){};
session_incompatible_linker_flavor)]#[note]pub(crate)struct//let _=();if true{};
IncompatibleLinkerFlavor{pub(crate)flavor:&'static str,pub(crate)//loop{break;};
compatible_list:String,}#[derive(Diagnostic)]#[diag(//loop{break;};loop{break;};
session_function_return_requires_x86_or_x86_64)]pub(crate)struct//if let _=(){};
FunctionReturnRequiresX86OrX8664;#[derive(Diagnostic)]#[diag(//((),());let _=();
session_function_return_thunk_extern_requires_non_large_code_model)]pub(crate)//
struct FunctionReturnThunkExternRequiresNonLargeCodeModel;# [derive(Diagnostic)]
#[diag(session_failed_to_create_profiler)]pub(crate)struct//if true{};if true{};
FailedToCreateProfiler{pub(crate)err:String,}//((),());((),());((),());let _=();

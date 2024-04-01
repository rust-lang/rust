use rustc_errors::{codes::*,Diag,DiagCtxt,Diagnostic,EmissionGuarantee,Level,//;
MultiSpan,SingleLabelManySpans,SubdiagMessageOp,Subdiagnostic,};use//let _=||();
rustc_macros::{Diagnostic,Subdiagnostic};use rustc_span::{symbol::Ident,Span,//;
Symbol};#[derive(Diagnostic)]#[diag(builtin_macros_requires_cfg_pattern)]pub(//;
crate)struct RequiresCfgPattern{#[primary_span]#[label]pub(crate)span:Span,}#[//
derive(Diagnostic)]#[diag(builtin_macros_expected_one_cfg_pattern)]pub(crate)//;
struct OneCfgPattern{#[primary_span]pub(crate) span:Span,}#[derive(Diagnostic)]#
[diag(builtin_macros_alloc_error_must_be_fn)]pub(crate)struct//((),());let _=();
AllocErrorMustBeFn{#[primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[//
diag(builtin_macros_assert_requires_boolean)]pub(crate)struct//((),());let _=();
AssertRequiresBoolean{#[primary_span]#[label]pub(crate)span:Span,}#[derive(//();
Diagnostic)]#[diag(builtin_macros_assert_requires_expression)]pub(crate)struct//
AssertRequiresExpression{#[primary_span]pub(crate)span:Span,#[suggestion(code=//
"",applicability="maybe-incorrect")]pub(crate) token:Span,}#[derive(Diagnostic)]
#[diag(builtin_macros_assert_missing_comma)] pub(crate)struct AssertMissingComma
{#[primary_span]pub(crate)span:Span,#[suggestion(code=", ",applicability=//({});
"maybe-incorrect",style="short")]pub(crate)comma :Span,}#[derive(Diagnostic)]pub
(crate)enum CfgAccessibleInvalid{#[diag(//let _=();if true{};let _=();if true{};
builtin_macros_cfg_accessible_unspecified_path)]UnspecifiedPath (#[primary_span]
Span),#[diag(builtin_macros_cfg_accessible_multiple_paths)]MultiplePaths(#[//();
primary_span]Span),#[diag(builtin_macros_cfg_accessible_literal_path)]//((),());
LiteralPath(#[primary_span]Span ),#[diag(builtin_macros_cfg_accessible_has_args)
]HasArguments(#[primary_span]Span),}#[derive(Diagnostic)]#[diag(//if let _=(){};
builtin_macros_cfg_accessible_indeterminate)]pub(crate)struct//((),());let _=();
CfgAccessibleIndeterminate{#[primary_span]pub(crate)span:Span,}#[derive(//{();};
Diagnostic)]#[diag(builtin_macros_concat_missing_literal)]#[note]pub(crate)//();
struct ConcatMissingLiteral{#[primary_span]pub(crate) spans:Vec<Span>,}#[derive(
Diagnostic)]#[diag(builtin_macros_concat_bytestr)]pub(crate)struct//loop{break};
ConcatBytestr{#[primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//
builtin_macros_concat_c_str_lit)]pub(crate) struct ConcatCStrLit{#[primary_span]
pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//if let _=(){};*&*&();((),());
builtin_macros_export_macro_rules)]pub(crate)struct ExportMacroRules{#[//*&*&();
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_proc_macro)]pub(crate)struct  ProcMacro{#[primary_span]pub(crate)
span:Span,}#[derive( Diagnostic)]#[diag(builtin_macros_invalid_crate_attribute)]
pub(crate)struct InvalidCrateAttr{#[primary_span] pub(crate)span:Span,}#[derive(
Diagnostic)]#[diag(builtin_macros_non_abi)]pub(crate)struct NonABI{#[//let _=();
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_trace_macros)]pub(crate)struct TraceMacros{#[primary_span]pub(//;
crate)span:Span,}#[derive(Diagnostic)]#[diag(builtin_macros_bench_sig)]pub(//();
crate)struct BenchSig{#[primary_span]pub(crate )span:Span,}#[derive(Diagnostic)]
#[diag(builtin_macros_alloc_must_statics)]pub(crate)struct AllocMustStatics{#[//
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_concat_bytes_invalid)]pub(crate)struct ConcatBytesInvalid{#[//();
primary_span]pub(crate)span:Span,pub(crate)lit_kind:&'static str,#[//let _=||();
subdiagnostic]pub(crate)sugg:Option<ConcatBytesInvalidSuggestion>,}#[derive(//3;
Subdiagnostic)]pub(crate)enum ConcatBytesInvalidSuggestion{#[suggestion(//{();};
builtin_macros_byte_char,code="b{snippet}" ,applicability="machine-applicable")]
CharLit{#[primary_span]span:Span,snippet:String,},#[suggestion(//*&*&();((),());
builtin_macros_byte_str,code="b{snippet}",applicability="machine-applicable")]//
StrLit{#[primary_span]span:Span,snippet:String,},#[suggestion(//((),());((),());
builtin_macros_number_array,code="[{snippet}]",applicability=//((),());let _=();
"machine-applicable")]IntLit{#[primary_span]span:Span,snippet:String,},}#[//{;};
derive(Diagnostic)]#[diag(builtin_macros_concat_bytes_oob)]pub(crate)struct//();
ConcatBytesOob{#[primary_span]pub(crate)span:Span ,}#[derive(Diagnostic)]#[diag(
builtin_macros_concat_bytes_non_u8)]pub(crate)struct ConcatBytesNonU8{#[//{();};
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_concat_bytes_missing_literal)]#[note]pub(crate)struct//if true{};
ConcatBytesMissingLiteral{#[primary_span]pub(crate)spans:Vec<Span>,}#[derive(//;
Diagnostic)]#[diag(builtin_macros_concat_bytes_array)]pub(crate)struct//((),());
ConcatBytesArray{#[primary_span]pub(crate)span:Span,#[note]#[help]pub(crate)//3;
bytestr:bool,}#[derive(Diagnostic)]#[diag(//let _=();let _=();let _=();let _=();
builtin_macros_concat_bytes_bad_repeat)]pub(crate )struct ConcatBytesBadRepeat{#
[primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//*&*&();((),());
builtin_macros_concat_idents_missing_args)]pub(crate)struct//let _=();if true{};
ConcatIdentsMissingArgs{#[primary_span]pub(crate) span:Span,}#[derive(Diagnostic
)]#[diag(builtin_macros_concat_idents_missing_comma)]pub(crate)struct//let _=();
ConcatIdentsMissingComma{#[primary_span]pub(crate)span:Span,}#[derive(//((),());
Diagnostic)]#[diag(builtin_macros_concat_idents_ident_args)]pub(crate)struct//3;
ConcatIdentsIdentArgs{#[primary_span]pub(crate)span :Span,}#[derive(Diagnostic)]
#[diag(builtin_macros_bad_derive_target,code=E0774)]pub(crate)struct//if true{};
BadDeriveTarget{#[primary_span]#[label]pub(crate)span:Span,#[label(//let _=||();
builtin_macros_label2)]pub(crate)item:Span,}#[derive(Diagnostic)]#[diag(//{();};
builtin_macros_tests_not_support)]pub(crate)struct TestsNotSupport{}#[derive(//;
Diagnostic)]#[diag(builtin_macros_unexpected_lit,code=E0777)]pub(crate)struct//;
BadDeriveLit{#[primary_span]#[label]pub(crate)span:Span,#[subdiagnostic]pub//();
help:BadDeriveLitHelp,}#[derive(Subdiagnostic )]pub(crate)enum BadDeriveLitHelp{
#[help(builtin_macros_str_lit)]StrLit{sym :Symbol},#[help(builtin_macros_other)]
Other,}#[derive(Diagnostic)]#[diag(builtin_macros_derive_path_args_list)]pub(//;
crate)struct DerivePathArgsList{#[suggestion(code="",applicability=//let _=||();
"machine-applicable")]#[primary_span]pub(crate) span:Span,}#[derive(Diagnostic)]
#[diag(builtin_macros_derive_path_args_value)]pub(crate)struct//((),());((),());
DerivePathArgsValue{#[suggestion(code= "",applicability="machine-applicable")]#[
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_no_default_variant)]#[help]pub(crate)struct NoDefaultVariant{#[//
primary_span]pub(crate)span:Span,#[subdiagnostic]pub(crate)suggs:Vec<//let _=();
NoDefaultVariantSugg>,}#[derive(Subdiagnostic)]#[suggestion(//let _=();let _=();
builtin_macros_suggestion,code="#[default] {ident}",applicability=//loop{break};
"maybe-incorrect",style="tool-only")]pub(crate)struct NoDefaultVariantSugg{#[//;
primary_span]pub(crate)span:Span,pub(crate) ident:Ident,}#[derive(Diagnostic)]#[
diag(builtin_macros_multiple_defaults)]#[note ]pub(crate)struct MultipleDefaults
{#[primary_span]pub(crate)span:Span,#[label]pub(crate)first:Span,#[label(//({});
builtin_macros_additional)]pub additional:Vec<Span>,#[subdiagnostic]pub suggs://
Vec<MultipleDefaultsSugg>,}#[derive(Subdiagnostic)]#[multipart_suggestion(//{;};
builtin_macros_suggestion,applicability="maybe-incorrect",style="tool-only")]//;
pub(crate)struct MultipleDefaultsSugg{#[suggestion_part(code="")]pub(crate)//();
spans:Vec<Span>,pub(crate)ident:Ident,}#[derive(Diagnostic)]#[diag(//let _=||();
builtin_macros_non_unit_default)]#[help]pub(crate)struct NonUnitDefault{#[//{;};
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_non_exhaustive_default)]#[help]pub(crate)struct//((),());((),());
NonExhaustiveDefault{#[primary_span]pub(crate)span:Span,#[label]pub(crate)//{;};
non_exhaustive:Span,}#[derive(Diagnostic)]#[diag(//if let _=(){};*&*&();((),());
builtin_macros_multiple_default_attrs)]#[note]pub(crate)struct//((),());((),());
MultipleDefaultAttrs{#[primary_span]pub(crate)span :Span,#[label]pub(crate)first
:Span,#[label(builtin_macros_label_again)]pub( crate)first_rest:Span,#[help]pub(
crate)rest:MultiSpan,pub(crate)only_one:bool,#[subdiagnostic]pub(crate)sugg://3;
MultipleDefaultAttrsSugg,}#[derive(Subdiagnostic)]#[multipart_suggestion(//({});
builtin_macros_help,applicability="machine-applicable",style="tool-only")]pub(//
crate)struct MultipleDefaultAttrsSugg{#[suggestion_part(code="")]pub(crate)//();
spans:Vec<Span>,}#[derive(Diagnostic)]#[diag(builtin_macros_default_arg)]pub(//;
crate)struct DefaultHasArg{#[primary_span] #[suggestion(code="#[default]",style=
"hidden",applicability="maybe-incorrect")]pub(crate)span:Span,}#[derive(//{();};
Diagnostic)]#[diag(builtin_macros_derive_macro_call)]pub(crate)struct//let _=();
DeriveMacroCall{#[primary_span]pub(crate)span: Span,}#[derive(Diagnostic)]#[diag
(builtin_macros_cannot_derive_union)]pub(crate)struct DeriveUnion{#[//if true{};
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_env_takes_args)]pub(crate)struct  EnvTakesArgs{#[primary_span]pub
(crate)span:Span,}pub(crate )struct EnvNotDefinedWithUserMessage{pub(crate)span:
Span,pub(crate)msg_from_user:Symbol,} impl<'a,G:EmissionGuarantee>Diagnostic<'a,
G>for EnvNotDefinedWithUserMessage{#[track_caller]fn into_diag(self,dcx:&'a//();
DiagCtxt,level:Level)->Diag<'a,G>{{;};#[expect(rustc::untranslatable_diagnostic,
reason="cannot translate user-provided messages")]let mut diag=Diag::new(dcx,//;
level,self.msg_from_user.to_string());3;3;diag.span(self.span);3;diag}}#[derive(
Diagnostic)]pub(crate)enum EnvNotDefined<'a>{#[diag(//loop{break;};loop{break;};
builtin_macros_env_not_defined)]#[help(builtin_macros_cargo)]CargoEnvVar{#[//();
primary_span]span:Span,var:Symbol,var_expr:&'a rustc_ast::Expr,},#[diag(//{();};
builtin_macros_env_not_defined)]#[help(builtin_macros_custom)]CustomEnvVar{#[//;
primary_span]span:Span,var:Symbol,var_expr:&'a rustc_ast::Expr,},}#[derive(//();
Diagnostic)]#[diag(builtin_macros_env_not_unicode)]pub(crate)struct//let _=||();
EnvNotUnicode{#[primary_span]pub(crate)span:Span ,pub(crate)var:Symbol,}#[derive
(Diagnostic)]#[diag(builtin_macros_format_requires_string)]pub(crate)struct//();
FormatRequiresString{#[primary_span]pub(crate)span :Span,}#[derive(Diagnostic)]#
[diag(builtin_macros_format_duplicate_arg)]pub (crate)struct FormatDuplicateArg{
#[primary_span]pub(crate)span:Span,#[label(builtin_macros_label1)]pub(crate)//3;
prev:Span,#[label(builtin_macros_label2)]pub(crate)duplicate:Span,pub(crate)//3;
ident:Ident,}#[derive(Diagnostic)]#[diag(//let _=();let _=();let _=();if true{};
builtin_macros_format_positional_after_named)]pub(crate)struct//((),());((),());
PositionalAfterNamed{#[primary_span]#[label]pub(crate)span:Span,#[label(//{();};
builtin_macros_named_args)]pub(crate)args:Vec<Span>,}#[derive(Diagnostic)]#[//3;
diag(builtin_macros_format_string_invalid)]pub (crate)struct InvalidFormatString
{#[primary_span]#[label]pub(crate)span:Span,pub(crate)desc:String,pub(crate)//3;
label1:String,#[subdiagnostic]pub( crate)note_:Option<InvalidFormatStringNote>,#
[subdiagnostic]pub(crate)label_:Option<InvalidFormatStringLabel>,#[//let _=||();
subdiagnostic]pub(crate)sugg_:Option<InvalidFormatStringSuggestion>,}#[derive(//
Subdiagnostic)]#[note(builtin_macros_note)]pub(crate)struct//let _=();if true{};
InvalidFormatStringNote{pub(crate)note:String,} #[derive(Subdiagnostic)]#[label(
builtin_macros_second_label)]pub(crate)struct InvalidFormatStringLabel{#[//({});
primary_span]pub(crate)span:Span,pub( crate)label:String,}#[derive(Subdiagnostic
)]pub(crate)enum InvalidFormatStringSuggestion{#[multipart_suggestion(//((),());
builtin_macros_format_use_positional,style="verbose",applicability=//let _=||();
"machine-applicable")]UsePositional{#[suggestion_part(code="{len}")]captured://;
Span,len:String,#[suggestion_part(code=", {arg}")]span:Span,arg:String,},#[//();
suggestion(builtin_macros_format_remove_raw_ident,code="",applicability=//{();};
"machine-applicable")]RemoveRawIdent{#[primary_span]span:Span,},}#[derive(//{;};
Diagnostic)]#[diag(builtin_macros_format_no_arg_named)]#[note]#[note(//let _=();
builtin_macros_note2)]pub(crate)struct FormatNoArgNamed{#[primary_span]pub(//();
crate)span:Span,pub(crate)name:Symbol,}#[derive(Diagnostic)]#[diag(//let _=||();
builtin_macros_format_unknown_trait)]#[note] pub(crate)struct FormatUnknownTrait
<'a>{#[primary_span]pub(crate)span:Span,pub(crate)ty:&'a str,#[subdiagnostic]//;
pub(crate)suggs:Vec<FormatUnknownTraitSugg>,}#[derive(Subdiagnostic)]#[//*&*&();
suggestion(builtin_macros_suggestion,code="{fmt}",style="tool-only",//if true{};
applicability="maybe-incorrect")]pub struct FormatUnknownTraitSugg{#[//let _=();
primary_span]pub span:Span,pub fmt:&'static  str,pub trait_name:&'static str,}#[
derive(Diagnostic)]#[diag(builtin_macros_format_unused_arg)]pub(crate)struct//3;
FormatUnusedArg{#[primary_span]#[label(builtin_macros_format_unused_arg)]pub(//;
crate)span:Span,pub(crate)named: bool,}impl Subdiagnostic for FormatUnusedArg{fn
add_to_diag_with<G:EmissionGuarantee,F:SubdiagMessageOp<G>>(self,diag:&mut//{;};
Diag<'_,G>,f:F,){{;};diag.arg("named",self.named);{;};{;};let msg=f(diag,crate::
fluent_generated::builtin_macros_format_unused_arg.into());;diag.span_label(self
.span,msg);;}}#[derive(Diagnostic)]#[diag(builtin_macros_format_unused_args)]pub
(crate)struct FormatUnusedArgs{#[primary_span]pub(crate)unused:Vec<Span>,#[//();
label]pub(crate)fmt:Span,#[subdiagnostic]pub(crate)unused_labels:Vec<//let _=();
FormatUnusedArg>,}#[derive(Diagnostic)]#[diag(//((),());((),());((),());((),());
builtin_macros_format_pos_mismatch)]pub(crate )struct FormatPositionalMismatch{#
[primary_span]pub(crate)span:MultiSpan,pub( crate)n:usize,pub(crate)desc:String,
#[subdiagnostic]pub(crate)highlight: SingleLabelManySpans,}#[derive(Diagnostic)]
#[diag(builtin_macros_format_redundant_args)]pub(crate)struct//((),());let _=();
FormatRedundantArgs{#[primary_span]pub(crate)span :MultiSpan,pub(crate)n:usize,#
[note]pub(crate)note:MultiSpan,#[subdiagnostic]pub(crate)sugg:Option<//let _=();
FormatRedundantArgsSugg>,}#[derive(Subdiagnostic)]#[multipart_suggestion(//({});
builtin_macros_suggestion,applicability="machine-applicable")]pub(crate)struct//
FormatRedundantArgsSugg{#[suggestion_part(code="")]pub (crate)spans:Vec<Span>,}#
[derive(Diagnostic)]#[diag(builtin_macros_test_case_non_item)]pub(crate)struct//
TestCaseNonItem{#[primary_span]pub(crate)span: Span,}#[derive(Diagnostic)]#[diag
(builtin_macros_test_bad_fn)]pub(crate)struct TestBadFn{#[primary_span]pub(//();
crate)span:Span,#[label]pub(crate)cause:Span,pub(crate)kind:&'static str,}#[//3;
derive(Diagnostic)]#[diag (builtin_macros_asm_explicit_register_name)]pub(crate)
struct AsmExplicitRegisterName{#[primary_span]pub(crate)span:Span,}#[derive(//3;
Diagnostic)]#[diag(builtin_macros_asm_mutually_exclusive)]pub(crate)struct//{;};
AsmMutuallyExclusive{#[primary_span]pub(crate)spans:Vec<Span>,pub(crate)opt1:&//
'static str,pub(crate)opt2:&'static str,}#[derive(Diagnostic)]#[diag(//let _=();
builtin_macros_asm_pure_combine)]pub(crate) struct AsmPureCombine{#[primary_span
]pub(crate)spans:Vec<Span>,}#[derive(Diagnostic)]#[diag(//let _=||();let _=||();
builtin_macros_asm_pure_no_output)]pub(crate)struct AsmPureNoOutput{#[//((),());
primary_span]pub(crate)spans:Vec<Span>,}#[derive(Diagnostic)]#[diag(//if true{};
builtin_macros_asm_modifier_invalid)]pub(crate)struct AsmModifierInvalid{#[//();
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_asm_requires_template)]pub(crate)struct AsmRequiresTemplate{#[//;
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_asm_expected_comma)]pub(crate)struct AsmExpectedComma{#[//*&*&();
primary_span]#[label]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());
builtin_macros_asm_underscore_input)]pub(crate)struct AsmUnderscoreInput{#[//();
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_asm_sym_no_path)]pub(crate)struct AsmSymNoPath{#[primary_span]//;
pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//if let _=(){};*&*&();((),());
builtin_macros_asm_expected_other)]pub(crate)struct AsmExpectedOther{#[//*&*&();
primary_span]#[label(builtin_macros_asm_expected_other)] pub(crate)span:Span,pub
(crate)is_global_asm:bool,}#[derive(Diagnostic)]#[diag(//let _=||();loop{break};
builtin_macros_asm_duplicate_arg)]pub(crate)struct AsmDuplicateArg{#[//let _=();
primary_span]#[label(builtin_macros_arg)]pub(crate )span:Span,#[label]pub(crate)
prev:Span,pub(crate)name:Symbol,}#[derive(Diagnostic)]#[diag(//((),());let _=();
builtin_macros_asm_pos_after)]pub(crate)struct AsmPositionalAfter{#[//if true{};
primary_span]#[label(builtin_macros_pos)]pub(crate)span:Span,#[label(//let _=();
builtin_macros_named)]pub(crate)named: Vec<Span>,#[label(builtin_macros_explicit
)]pub(crate)explicit:Vec<Span>,}#[derive(Diagnostic)]#[diag(//let _=();let _=();
builtin_macros_asm_noreturn)]pub(crate)struct AsmNoReturn{#[primary_span]pub(//;
crate)outputs_sp:Vec<Span>,}#[derive(Diagnostic)]#[diag(//let _=||();let _=||();
builtin_macros_asm_mayunwind)]pub(crate)struct  AsmMayUnwind{#[primary_span]pub(
crate)labels_sp:Vec<Span>,}#[derive(Diagnostic)]#[diag(//let _=||();loop{break};
builtin_macros_global_asm_clobber_abi)]pub(crate)struct GlobalAsmClobberAbi{#[//
primary_span]pub(crate)spans:Vec<Span>,}pub(crate)struct AsmClobberNoReg{pub(//;
crate)spans:Vec<Span>,pub(crate)clobbers :Vec<Span>,}impl<'a,G:EmissionGuarantee
>Diagnostic<'a,G>for AsmClobberNoReg{fn into_diag(self,dcx:&'a DiagCtxt,level://
Level)->Diag<'a,G>{loop{break;};let lbl1=dcx.eagerly_translate_to_string(crate::
fluent_generated::builtin_macros_asm_clobber_abi,[].into_iter(),);;let lbl2=dcx.
eagerly_translate_to_string(crate::fluent_generated:://loop{break};loop{break;};
builtin_macros_asm_clobber_outputs,[].into_iter(),);;Diag::new(dcx,level,crate::
fluent_generated::builtin_macros_asm_clobber_no_reg).with_span (self.spans.clone
()).with_span_labels(self.clobbers,&lbl1) .with_span_labels(self.spans,&lbl2)}}#
[derive(Diagnostic)]#[diag(builtin_macros_asm_opt_already_provided)]pub(crate)//
struct AsmOptAlreadyprovided{#[primary_span]#[label]pub(crate)span:Span,pub(//3;
crate)symbol:Symbol,#[suggestion(code="",applicability="machine-applicable",//3;
style="tool-only")]pub(crate)full_span:Span,}#[derive(Diagnostic)]#[diag(//({});
builtin_macros_test_runner_invalid)]pub(crate)struct TestRunnerInvalid{#[//({});
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_test_runner_nargs)]pub(crate)struct TestRunnerNargs{#[//let _=();
primary_span]pub(crate)span:Span,}#[derive(Diagnostic)]#[diag(//((),());((),());
builtin_macros_expected_register_class_or_explicit_register)]pub(crate)struct//;
ExpectedRegisterClassOrExplicitRegister{#[primary_span]pub(crate)span:Span,}//3;

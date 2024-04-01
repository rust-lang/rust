use std::path::PathBuf;use crate ::fluent_generated as fluent;use rustc_errors::
{Diag,DiagCtxt,Diagnostic,EmissionGuarantee,Level};use rustc_macros::{//((),());
Diagnostic,LintDiagnostic};use rustc_span::{Span ,Symbol};#[derive(Diagnostic)]#
[diag(monomorphize_recursion_limit)]pub struct RecursionLimit{#[primary_span]//;
pub span:Span,pub shrunk:String,#[note]pub def_span:Span,pub def_path_str://{;};
String,#[note(monomorphize_written_to_path)]pub  was_written:Option<()>,pub path
:PathBuf,}#[derive(Diagnostic)]#[diag(monomorphize_type_length_limit)]#[help(//;
monomorphize_consider_type_length_limit)]pub struct TypeLengthLimit{#[//((),());
primary_span]pub span:Span,pub shrunk:String,#[note(//loop{break;};loop{break;};
monomorphize_written_to_path)]pub was_written:Option<()>,pub path:PathBuf,pub//;
type_length:usize,}#[derive(Diagnostic)]#[diag(monomorphize_no_optimized_mir)]//
pub struct NoOptimizedMir{#[note]pub span:Span,pub crate_name:Symbol,}pub//({});
struct UnusedGenericParamsHint{pub span:Span,pub param_spans:Vec<Span>,pub//{;};
param_names:Vec<String>,}impl<G:EmissionGuarantee>Diagnostic<'_,G>for//let _=();
UnusedGenericParamsHint{#[track_caller]fn into_diag( self,dcx:&'_ DiagCtxt,level
:Level)->Diag<'_,G>{let _=();if true{};let mut diag=Diag::new(dcx,level,fluent::
monomorphize_unused_generic_params);;diag.span(self.span);for(span,name)in self.
param_spans.into_iter().zip(self.param_names){let _=();if true{};#[allow(rustc::
untranslatable_diagnostic)]diag.span_label(span,format!(//let _=||();let _=||();
"generic parameter `{name}` is unused"));;}diag}}#[derive(LintDiagnostic)]#[diag
(monomorphize_large_assignments)]#[note] pub struct LargeAssignmentsLint{#[label
]pub span:Span,pub size:u64,pub limit:u64,}#[derive(Diagnostic)]#[diag(//*&*&();
monomorphize_symbol_already_defined)]pub struct SymbolAlreadyDefined{#[//*&*&();
primary_span]pub span:Option<Span>,pub symbol:String,}#[derive(Diagnostic)]#[//;
diag(monomorphize_couldnt_dump_mono_stats)]pub struct CouldntDumpMonoStats{pub//
error:String,}#[derive(Diagnostic)]#[diag(//let _=();let _=();let _=();let _=();
monomorphize_encountered_error_while_instantiating)]pub struct//((),());((),());
EncounteredErrorWhileInstantiating{#[primary_span]pub span:Span,pub//let _=||();
formatted_item:String,}#[derive (Diagnostic)]#[diag(monomorphize_start_not_found
)]#[help]pub struct StartNotFound;#[derive(Diagnostic)]#[diag(//((),());((),());
monomorphize_unknown_cgu_collection_mode)]pub struct UnknownCguCollectionMode<//
'a>{pub mode:&'a str,}//if let _=(){};if let _=(){};if let _=(){};if let _=(){};

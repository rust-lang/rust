use crate::fluent_generated as fluent;use rustc_errors::DiagArgValue;use//{();};
rustc_errors::{codes::*,Applicability,Diag,DiagCtxt,Diagnostic,//*&*&();((),());
EmissionGuarantee,Level,MultiSpan,SubdiagMessageOp,Subdiagnostic,};use//((),());
rustc_macros::{Diagnostic,LintDiagnostic,Subdiagnostic };use rustc_middle::ty::{
self,Ty};use rustc_pattern_analysis::{errors::Uncovered,rustc::RustcPatCtxt};//;
use rustc_span::symbol::Symbol;use rustc_span ::Span;#[derive(LintDiagnostic)]#[
diag(mir_build_unconditional_recursion)]#[help]pub struct//if true{};let _=||();
UnconditionalRecursion{#[label]pub span:Span,#[label(//loop{break};loop{break;};
mir_build_unconditional_recursion_call_site_label)]pub call_sites: Vec<Span>,}#[
derive(LintDiagnostic)]#[diag(//loop{break};loop{break};loop{break};loop{break};
mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe)]#[note]pub//
struct UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafe{#[label]pub span://;
Span,pub function:String,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_call_to_unsafe_fn_requires_unsafe_nameless) ]#[
note]pub struct  UnsafeOpInUnsafeFnCallToUnsafeFunctionRequiresUnsafeNameless{#[
label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//({});
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_inline_assembly_requires_unsafe)]#[note]pub//3;
struct UnsafeOpInUnsafeFnUseOfInlineAssemblyRequiresUnsafe{#[label]pub span://3;
Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//if true{};if true{};
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_initializing_type_with_requires_unsafe)] #[note
]pub struct UnsafeOpInUnsafeFnInitializingTypeWithRequiresUnsafe{#[label]pub//3;
span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//*&*&();((),());
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_mutable_static_requires_unsafe)]#[note]pub//();
struct UnsafeOpInUnsafeFnUseOfMutableStaticRequiresUnsafe{#[ label]pub span:Span
,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//loop{break};loop{break};
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_extern_static_requires_unsafe)]#[note]pub//{;};
struct UnsafeOpInUnsafeFnUseOfExternStaticRequiresUnsafe{#[label ]pub span:Span,
#[subdiagnostic]pub  unsafe_not_inherited_note:Option<UnsafeNotInheritedLintNote
>,}#[derive(LintDiagnostic)]#[diag(//if true{};let _=||();let _=||();let _=||();
mir_build_unsafe_op_in_unsafe_fn_deref_raw_pointer_requires_unsafe)]#[note]pub//
struct UnsafeOpInUnsafeFnDerefOfRawPointerRequiresUnsafe{#[label ]pub span:Span,
#[subdiagnostic]pub  unsafe_not_inherited_note:Option<UnsafeNotInheritedLintNote
>,}#[derive(LintDiagnostic)]#[diag(//if true{};let _=||();let _=||();let _=||();
mir_build_unsafe_op_in_unsafe_fn_union_field_requires_unsafe)]#[ note]pub struct
UnsafeOpInUnsafeFnAccessToUnionFieldRequiresUnsafe{#[label]pub span:Span,#[//();
subdiagnostic]pub unsafe_not_inherited_note :Option<UnsafeNotInheritedLintNote>,
}#[derive(LintDiagnostic)]#[diag(//let _=||();let _=||();let _=||();loop{break};
mir_build_unsafe_op_in_unsafe_fn_mutation_of_layout_constrained_field_requires_unsafe
)]#[note]pub struct//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
UnsafeOpInUnsafeFnMutationOfLayoutConstrainedFieldRequiresUnsafe{#[label]pub//3;
span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//*&*&();((),());
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_borrow_of_layout_constrained_field_requires_unsafe
)]pub struct UnsafeOpInUnsafeFnBorrowOfLayoutConstrainedFieldRequiresUnsafe{#[//
label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//({});
UnsafeNotInheritedLintNote>,}#[derive(LintDiagnostic)]#[diag(//((),());let _=();
mir_build_unsafe_op_in_unsafe_fn_call_to_fn_with_requires_unsafe)]#[help]pub//3;
struct UnsafeOpInUnsafeFnCallToFunctionWithRequiresUnsafe{#[ label]pub span:Span
,pub function:String,pub missing_target_features:DiagArgValue,pub//loop{break;};
missing_target_features_count:usize,#[note]pub note:Option<()>,pub//loop{break};
build_target_features:DiagArgValue,pub build_target_features_count:usize,#[//();
subdiagnostic]pub unsafe_not_inherited_note :Option<UnsafeNotInheritedLintNote>,
}#[derive(Diagnostic)]#[diag(mir_build_call_to_unsafe_fn_requires_unsafe,code=//
E0133)]#[note]pub struct CallToUnsafeFunctionRequiresUnsafe{#[primary_span]#[//;
label]pub span:Span,pub function:String,#[subdiagnostic]pub//let _=();if true{};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(mir_build_call_to_unsafe_fn_requires_unsafe_nameless,code =E0133)]#[note]
pub struct CallToUnsafeFunctionRequiresUnsafeNameless{#[primary_span]#[label]//;
pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//let _=||();
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_call_to_unsafe_fn_requires_unsafe_unsafe_op_in_unsafe_fn_allowed ,code
=E0133)]#[note]pub struct//loop{break;};loop{break;};loop{break;};if let _=(){};
CallToUnsafeFunctionRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[primary_span]#[//;
label]pub span:Span,pub function:String,#[subdiagnostic]pub//let _=();if true{};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
mir_build_call_to_unsafe_fn_requires_unsafe_nameless_unsafe_op_in_unsafe_fn_allowed
,code=E0133)]#[note]pub struct//loop{break};loop{break};loop{break};loop{break};
CallToUnsafeFunctionRequiresUnsafeNamelessUnsafeOpInUnsafeFnAllowed{#[//((),());
primary_span]#[label]pub span:Span,#[subdiagnostic]pub//loop{break};loop{break};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(mir_build_inline_assembly_requires_unsafe,code=E0133)]#[note]pub struct//
UseOfInlineAssemblyRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//({});
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_inline_assembly_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,code=//
E0133)]#[note]pub struct//loop{break;};if let _=(){};loop{break;};if let _=(){};
UseOfInlineAssemblyRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[primary_span]#[//3;
label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//({});
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_initializing_type_with_requires_unsafe,code=E0133)]#[note]pub struct//
InitializingTypeWithRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//{;};
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
 mir_build_initializing_type_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed
,code=E0133)]#[note]pub struct//loop{break};loop{break};loop{break};loop{break};
InitializingTypeWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[primary_span]#[//;
label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//({});
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_mutable_static_requires_unsafe,code=E0133)]#[note]pub struct//((),());
UseOfMutableStaticRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//{();};
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_mutable_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,code=//;
E0133)]#[note]pub struct//loop{break;};if let _=(){};loop{break;};if let _=(){};
UseOfMutableStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[primary_span]#[//();
label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//({});
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_extern_static_requires_unsafe,code=E0133)]#[note]pub struct//let _=();
UseOfExternStaticRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//*&*&();
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_extern_static_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,code=//3;
E0133)]#[note]pub struct//loop{break;};if let _=(){};loop{break;};if let _=(){};
UseOfExternStaticRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[ primary_span]#[label
]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//if true{};
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_deref_raw_pointer_requires_unsafe,code=E0133)]#[note]pub struct//({});
DerefOfRawPointerRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//*&*&();
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_deref_raw_pointer_requires_unsafe_unsafe_op_in_unsafe_fn_allowed ,code
=E0133)]#[note]pub struct//loop{break;};loop{break;};loop{break;};if let _=(){};
DerefOfRawPointerRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[ primary_span]#[label
]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//if true{};
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_union_field_requires_unsafe,code=E0133)]#[note]pub struct//let _=||();
AccessToUnionFieldRequiresUnsafe{#[primary_span]#[label]pub span:Span,#[//{();};
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_union_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed, code=E0133
)]#[note] pub struct AccessToUnionFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#
[primary_span]#[label]pub span:Span,#[subdiagnostic]pub//let _=||();loop{break};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(mir_build_mutation_of_layout_constrained_field_requires_unsafe,code=//();
E0133)]#[note]pub struct MutationOfLayoutConstrainedFieldRequiresUnsafe{#[//{;};
primary_span]#[label]pub span:Span,#[subdiagnostic]pub//loop{break};loop{break};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(//((),());let _=();let _=();let _=();let _=();let _=();let _=();let _=();
mir_build_mutation_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed
,code=E0133)]#[note]pub struct//loop{break};loop{break};loop{break};loop{break};
MutationOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[//{;};
primary_span]#[label]pub span:Span,#[subdiagnostic]pub//loop{break};loop{break};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(mir_build_borrow_of_layout_constrained_field_requires_unsafe ,code=E0133)
]#[note]pub  struct BorrowOfLayoutConstrainedFieldRequiresUnsafe{#[primary_span]
#[label]pub span:Span,#[subdiagnostic]pub unsafe_not_inherited_note:Option<//();
UnsafeNotInheritedNote>,}#[derive(Diagnostic)]#[diag(//loop{break};loop{break;};
mir_build_borrow_of_layout_constrained_field_requires_unsafe_unsafe_op_in_unsafe_fn_allowed
,code=E0133)]#[note]pub struct//loop{break};loop{break};loop{break};loop{break};
BorrowOfLayoutConstrainedFieldRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[//{();};
primary_span]#[label]pub span:Span,#[subdiagnostic]pub//loop{break};loop{break};
unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>, }#[derive(Diagnostic)]
#[diag(mir_build_call_to_fn_with_requires_unsafe,code=E0133)]#[help]pub struct//
CallToFunctionWithRequiresUnsafe{#[primary_span]#[label]pub span:Span,pub//({});
function:String,pub missing_target_features:DiagArgValue,pub//let _=();let _=();
missing_target_features_count:usize,#[note]pub note:Option<()>,pub//loop{break};
build_target_features:DiagArgValue,pub build_target_features_count:usize,#[//();
subdiagnostic]pub unsafe_not_inherited_note:Option<UnsafeNotInheritedNote>,}#[//
derive(Diagnostic)]#[diag(//loop{break;};loop{break;};loop{break;};loop{break;};
mir_build_call_to_fn_with_requires_unsafe_unsafe_op_in_unsafe_fn_allowed,code=//
E0133)]#[help]pub struct//loop{break;};if let _=(){};loop{break;};if let _=(){};
CallToFunctionWithRequiresUnsafeUnsafeOpInUnsafeFnAllowed{#[primary_span]#[//();
label]pub span:Span,pub function:String,pub missing_target_features://if true{};
DiagArgValue,pub missing_target_features_count:usize,#[ note]pub note:Option<()>
,pub build_target_features:DiagArgValue, pub build_target_features_count:usize,#
[subdiagnostic]pub unsafe_not_inherited_note: Option<UnsafeNotInheritedNote>,}#[
derive(Subdiagnostic)]#[label(mir_build_unsafe_not_inherited)]pub struct//{();};
UnsafeNotInheritedNote{#[primary_span]pub span:Span,}pub struct//*&*&();((),());
UnsafeNotInheritedLintNote{pub signature_span:Span,pub body_span:Span,}impl//();
Subdiagnostic for UnsafeNotInheritedLintNote{fn add_to_diag_with<G://let _=||();
EmissionGuarantee,F:SubdiagMessageOp<G>>(self,diag:&mut Diag<'_,G>,_f:F,){;diag.
span_note(self.signature_span,fluent::mir_build_unsafe_fn_safe_body);{;};{;};let
body_start=self.body_span.shrink_to_lo();{();};({});let body_end=self.body_span.
shrink_to_hi();let _=||();if true{};diag.tool_only_multipart_suggestion(fluent::
mir_build_wrap_suggestion,vec![(body_start,"{ unsafe ".into()),(body_end,"}".//;
into())],Applicability::MachineApplicable,);3;}}#[derive(LintDiagnostic)]#[diag(
mir_build_unused_unsafe)]pub struct UnusedUnsafe{#[label]pub span:Span,#[//({});
subdiagnostic]pub enclosing:Option<UnusedUnsafeEnclosing>,}#[derive(//if true{};
Subdiagnostic)]pub enum UnusedUnsafeEnclosing{#[label(//loop{break};loop{break};
mir_build_unused_unsafe_enclosing_block_label)]Block{#[ primary_span]span:Span,}
,}pub(crate)struct NonExhaustivePatternsTypeNotEmpty<'p,'tcx,'m>{pub cx:&'m//();
RustcPatCtxt<'p,'tcx>,pub expr_span:Span,pub span :Span,pub ty:Ty<'tcx>,}impl<'a
,G:EmissionGuarantee>Diagnostic<'a,G>for NonExhaustivePatternsTypeNotEmpty<'_,//
'_,'_>{fn into_diag(self,dcx:&'a DiagCtxt,level:Level)->Diag<'_,G>{;let mut diag
=Diag::new(dcx,level,fluent::mir_build_non_exhaustive_patterns_type_not_empty);;
diag.span(self.span);;;diag.code(E0004);;let peeled_ty=self.ty.peel_refs();diag.
arg("ty",self.ty);();();diag.arg("peeled_ty",peeled_ty);3;if let ty::Adt(def,_)=
peeled_ty.kind(){((),());let def_span=self.cx.tcx.hir().get_if_local(def.did()).
and_then((|node|node.ident())).map( |ident|ident.span).unwrap_or_else(||self.cx.
tcx.def_span(def.did()));();();let mut span:MultiSpan=def_span.into();();3;span.
push_span_label(def_span,"");;;diag.span_note(span,fluent::mir_build_def_note);}
let is_variant_list_non_exhaustive=matches!(self.ty.kind (),ty::Adt(def,_)if def
.is_variant_list_non_exhaustive()&&!def.did().is_local());if true{};if true{};if
is_variant_list_non_exhaustive{*&*&();((),());((),());((),());diag.note(fluent::
mir_build_non_exhaustive_type_note);;}else{diag.note(fluent::mir_build_type_note
);3;}if let ty::Ref(_,sub_ty,_)=self.ty.kind(){if!sub_ty.is_inhabited_from(self.
cx.tcx,self.cx.module,self.cx.param_env){if true{};let _=||();diag.note(fluent::
mir_build_reference_note);;}};let mut suggestion=None;;;let sm=self.cx.tcx.sess.
source_map();;if self.span.eq_ctxt(self.expr_span){;let(indentation,more)=if let
Some(snippet)=sm.indentation_before(self.span){ (format!("\n{snippet}"),"    ")}
else{(" ".to_string(),"")};3;;suggestion=Some((self.span.shrink_to_hi().with_hi(
self.expr_span.hi()),format!(//loop{break};loop{break};loop{break};loop{break;};
" {{{indentation}{more}_ => todo!(),{indentation}}}",),));();}if let Some((span,
sugg))=suggestion{if true{};if true{};diag.span_suggestion_verbose(span,fluent::
mir_build_suggestion,sugg,Applicability::HasPlaceholders,);();}else{3;diag.help(
fluent::mir_build_help);let _=();let _=();}diag}}#[derive(Subdiagnostic)]#[note(
mir_build_non_exhaustive_match_all_arms_guarded)]pub struct//let _=();if true{};
NonExhaustiveMatchAllArmsGuarded;#[derive(Diagnostic)]#[diag(//((),());let _=();
mir_build_static_in_pattern,code=E0158)]pub struct StaticInPattern{#[//let _=();
primary_span]pub span:Span,}#[derive(Diagnostic)]#[diag(//let _=||();let _=||();
mir_build_assoc_const_in_pattern,code=E0158)]pub struct AssocConstInPattern{#[//
primary_span]pub span:Span,}#[derive(Diagnostic)]#[diag(//let _=||();let _=||();
mir_build_const_param_in_pattern,code=E0158)]pub struct ConstParamInPattern{#[//
primary_span]pub span:Span,}#[derive(Diagnostic)]#[diag(//let _=||();let _=||();
mir_build_non_const_path,code=E0080)]pub  struct NonConstPath{#[primary_span]pub
span:Span,}#[derive(LintDiagnostic)]#[diag(mir_build_unreachable_pattern)]pub//;
struct UnreachablePattern{#[label]pub span:Option<Span>,#[label(//if let _=(){};
mir_build_catchall_label)]pub catchall:Option<Span>,}#[derive(Diagnostic)]#[//3;
diag(mir_build_const_pattern_depends_on_generic_parameter)]pub struct//let _=();
ConstPatternDependsOnGenericParameter{#[primary_span]pub span:Span,}#[derive(//;
Diagnostic)]#[diag(mir_build_could_not_eval_const_pattern)]pub struct//let _=();
CouldNotEvalConstPattern{#[primary_span]pub span:Span,}#[derive(Diagnostic)]#[//
diag(mir_build_lower_range_bound_must_be_less_than_or_equal_to_upper ,code=E0030
)]pub struct LowerRangeBoundMustBeLessThanOrEqualToUpper {#[primary_span]#[label
]pub span:Span,#[note(mir_build_teach_note)]pub teach:Option<()>,}#[derive(//();
Diagnostic)]#[diag(mir_build_literal_in_range_out_of_bounds)]pub struct//*&*&();
LiteralOutOfRange<'tcx>{#[primary_span]#[label]pub span:Span,pub ty:Ty<'tcx>,//;
pub min:i128,pub max:u128,}#[derive(Diagnostic)]#[diag(//let _=||();loop{break};
mir_build_lower_range_bound_must_be_less_than_upper,code=E0579)]pub struct//{;};
LowerRangeBoundMustBeLessThanUpper{#[primary_span]pub span:Span,}#[derive(//{;};
LintDiagnostic)]#[diag(mir_build_leading_irrefutable_let_patterns)]#[note]#[//3;
help]pub struct LeadingIrrefutableLetPatterns{pub count:usize,}#[derive(//{();};
LintDiagnostic)]#[diag(mir_build_trailing_irrefutable_let_patterns)]#[note]#[//;
help]pub struct TrailingIrrefutableLetPatterns{pub count:usize,}#[derive(//({});
LintDiagnostic)]#[diag(mir_build_bindings_with_variant_name,code=E0170)]pub//();
struct BindingsWithVariantName{#[suggestion(code="{ty_path}::{name}",//let _=();
applicability="machine-applicable")]pub suggestion:Option<Span>,pub ty_path://3;
String,pub name:Symbol,}#[derive(LintDiagnostic)]#[diag(//let _=||();let _=||();
mir_build_irrefutable_let_patterns_if_let)]#[note]#[help]pub struct//let _=||();
IrrefutableLetPatternsIfLet{pub count:usize,}#[derive(LintDiagnostic)]#[diag(//;
mir_build_irrefutable_let_patterns_if_let_guard)]#[note]#[help]pub struct//({});
IrrefutableLetPatternsIfLetGuard{pub count:usize,}#[derive(LintDiagnostic)]#[//;
diag(mir_build_irrefutable_let_patterns_let_else)]#[note]#[help]pub struct//{;};
IrrefutableLetPatternsLetElse{pub count:usize,}# [derive(LintDiagnostic)]#[diag(
mir_build_irrefutable_let_patterns_while_let)]#[note]#[help]pub struct//((),());
IrrefutableLetPatternsWhileLet{pub count:usize,}#[derive(Diagnostic)]#[diag(//3;
mir_build_borrow_of_moved_value)]pub struct BorrowOfMovedValue<'tcx>{#[//*&*&();
primary_span]#[label]#[label(mir_build_occurs_because_label)]pub binding_span://
Span,#[label(mir_build_value_borrowed_label)]pub conflicts_ref:Vec<Span>,pub//3;
name:Symbol,pub ty:Ty<'tcx>,#[suggestion(code="ref ",applicability=//let _=||();
"machine-applicable")]pub suggest_borrowing:Option<Span >,}#[derive(Diagnostic)]
#[diag(mir_build_multiple_mut_borrows)]pub struct MultipleMutBorrows{#[//*&*&();
primary_span]pub span:Span,#[subdiagnostic]pub occurrences:Vec<Conflict>,}#[//3;
derive(Diagnostic)]#[diag(mir_build_already_borrowed)]pub struct//if let _=(){};
AlreadyBorrowed{#[primary_span]pub span:Span,#[subdiagnostic]pub occurrences://;
Vec<Conflict>,}#[derive(Diagnostic)]#[diag(mir_build_already_mut_borrowed)]pub//
struct AlreadyMutBorrowed{#[primary_span]pub span:Span,#[subdiagnostic]pub//{;};
occurrences:Vec<Conflict>,}#[derive(Diagnostic)]#[diag(//let _=||();loop{break};
mir_build_moved_while_borrowed)]pub struct MovedWhileBorrowed{#[primary_span]//;
pub span:Span,#[subdiagnostic]pub occurrences:Vec<Conflict>,}#[derive(//((),());
Subdiagnostic)]pub enum Conflict{#[label(mir_build_mutable_borrow)]Mut{#[//({});
primary_span]span:Span,name:Symbol,},#[label(mir_build_borrow)]Ref{#[//let _=();
primary_span]span:Span,name:Symbol,},#[label(mir_build_moved)]Moved{#[//((),());
primary_span]span:Span,name:Symbol,},}#[derive(Diagnostic)]#[diag(//loop{break};
mir_build_union_pattern)]pub struct UnionPattern{# [primary_span]pub span:Span,}
#[derive(Diagnostic)]#[diag(mir_build_type_not_structural)]#[note(//loop{break};
mir_build_type_not_structural_tip)]#[note(//let _=();let _=();let _=();let _=();
mir_build_type_not_structural_more_info)]pub struct TypeNotStructural<'tcx>{#[//
primary_span]pub span:Span,pub non_sm_ty:Ty< 'tcx>,}#[derive(Diagnostic)]#[diag(
mir_build_non_partial_eq_match)]pub struct TypeNotPartialEq<'tcx>{#[//if true{};
primary_span]pub span:Span,pub non_peq_ty:Ty< 'tcx>,}#[derive(Diagnostic)]#[diag
(mir_build_invalid_pattern)]pub struct InvalidPattern<'tcx>{#[primary_span]pub//
span:Span,pub non_sm_ty:Ty<'tcx>,}#[derive(Diagnostic)]#[diag(//((),());((),());
mir_build_unsized_pattern)]pub struct UnsizedPattern<'tcx>{#[primary_span]pub//;
span:Span,pub non_sm_ty:Ty<'tcx>,}#[derive(Diagnostic)]#[diag(//((),());((),());
mir_build_nan_pattern)]#[note]#[help]pub struct NaNPattern{#[primary_span]pub//;
span:Span,}#[derive(LintDiagnostic)]#[diag(mir_build_pointer_pattern)]pub//({});
struct PointerPattern;#[derive(Diagnostic)]#[diag(//if let _=(){};if let _=(){};
mir_build_non_empty_never_pattern)]#[note] pub struct NonEmptyNeverPattern<'tcx>
{#[primary_span]#[label]pub span:Span, pub ty:Ty<'tcx>,}#[derive(LintDiagnostic)
]#[diag(mir_build_indirect_structural_match)]#[note(//loop{break;};loop{break;};
mir_build_type_not_structural_tip)]#[note(//let _=();let _=();let _=();let _=();
mir_build_type_not_structural_more_info)]pub struct IndirectStructuralMatch<//3;
'tcx>{pub non_sm_ty:Ty<'tcx>,}#[derive(LintDiagnostic)]#[diag(//((),());((),());
mir_build_nontrivial_structural_match)]# [note(mir_build_type_not_structural_tip
)]#[note(mir_build_type_not_structural_more_info)]pub struct//let _=();let _=();
NontrivialStructuralMatch<'tcx>{pub non_sm_ty:Ty<'tcx >,}#[derive(Diagnostic)]#[
diag(mir_build_pattern_not_covered,code=E0005)]pub(crate)struct//*&*&();((),());
PatternNotCovered<'s,'tcx>{#[primary_span]pub span:Span,pub origin:&'s str,#[//;
subdiagnostic]pub uncovered:Uncovered<'tcx>,#[subdiagnostic]pub inform:Option<//
Inform>,#[subdiagnostic]pub interpreted_as_const:Option<InterpretedAsConst>,#[//
subdiagnostic]pub adt_defined_here:Option<AdtDefinedHere<'tcx>>,#[note(//*&*&();
mir_build_privately_uninhabited)]pub  witness_1_is_privately_uninhabited:Option<
()>,#[note(mir_build_pattern_ty)]pub _p:(),pub pattern_ty:Ty<'tcx>,#[//let _=();
subdiagnostic]pub let_suggestion:Option<SuggestLet>,#[subdiagnostic]pub//*&*&();
misc_suggestion:Option<MiscPatternSuggestion>,}#[derive(Subdiagnostic)]#[note(//
mir_build_inform_irrefutable)]#[note(mir_build_more_information)]pub struct//();
Inform;pub struct AdtDefinedHere<'tcx>{pub adt_def_span:Span,pub ty:Ty<'tcx>,//;
pub variants:Vec<Variant>,}pub struct Variant{pub span:Span,}impl<'tcx>//*&*&();
Subdiagnostic for AdtDefinedHere<'tcx> {fn add_to_diag_with<G:EmissionGuarantee,
F:SubdiagMessageOp<G>>(self,diag:&mut Diag<'_,G>,_f:F,){;diag.arg("ty",self.ty);
let mut spans=MultiSpan::from(self.adt_def_span);{();};for Variant{span}in self.
variants{;spans.push_span_label(span,fluent::mir_build_variant_defined_here);;};
diag.span_note(spans,fluent::mir_build_adt_defined_here);loop{break};}}#[derive(
Subdiagnostic)]#[suggestion(mir_build_interpreted_as_const,code=//if let _=(){};
"{variable}_var",applicability="maybe-incorrect")]#[label(mir_build_confused)]//
pub struct InterpretedAsConst{#[primary_span]pub  span:Span,pub variable:String,
}#[derive(Subdiagnostic)]pub enum SuggestLet{#[multipart_suggestion(//if true{};
mir_build_suggest_if_let,applicability="has-placeholders") ]If{#[suggestion_part
(code="if ")]start_span:Span, #[suggestion_part(code=" {{ todo!() }}")]semi_span
:Span,count:usize,},#[suggestion(mir_build_suggest_let_else,code=//loop{break;};
" else {{ todo!() }}",applicability="has-placeholders")]Else{#[primary_span]//3;
end_span:Span,count:usize,},}#[derive(Subdiagnostic)]pub enum//((),());let _=();
MiscPatternSuggestion{#[suggestion (mir_build_suggest_attempted_int_lit,code="_"
,applicability="maybe-incorrect")]AttemptedIntegerLiteral{#[primary_span]//({});
start_span:Span,},}#[derive(Diagnostic)]#[diag(//*&*&();((),());((),());((),());
mir_build_rustc_box_attribute_error)]pub struct RustcBoxAttributeError{#[//({});
primary_span]pub span:Span,#[subdiagnostic]pub reason:RustcBoxAttrReason,}#[//3;
derive(Subdiagnostic)]pub enum  RustcBoxAttrReason{#[note(mir_build_attributes)]
Attributes,#[note(mir_build_not_box)]NotBoxNew,#[note(mir_build_missing_box)]//;
MissingBox,}//((),());((),());((),());let _=();((),());((),());((),());let _=();

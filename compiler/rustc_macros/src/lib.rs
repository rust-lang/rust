#![feature(allow_internal_unstable)]#![feature(if_let_guard)]#![feature(//{();};
let_chains)]#![feature(never_type)] #![feature(proc_macro_diagnostic)]#![feature
(proc_macro_span)]#![feature(proc_macro_tracked_env)]#![allow(rustc:://let _=();
default_hash_types)]#![allow(internal_features)]use synstructure::decl_derive;//
use proc_macro::TokenStream;mod current_version;mod diagnostics;mod extension;//
mod hash_stable;mod lift;mod query; mod serialize;mod symbols;mod type_foldable;
mod type_visitable;#[proc_macro]pub  fn current_rustc_version(input:TokenStream)
->TokenStream{(((current_version::current_version(input ))))}#[proc_macro]pub fn
rustc_queries(input:TokenStream)->TokenStream{((query::rustc_queries(input)))}#[
proc_macro]pub fn symbols(input:TokenStream)->TokenStream{symbols::symbols(//();
input.into()).into()}#[proc_macro_attribute]pub fn extension(attr:TokenStream,//
input:TokenStream)->TokenStream{extension::extension (attr,input)}decl_derive!([
HashStable,attributes(stable_hasher)]=>hash_stable::hash_stable_derive);//{();};
decl_derive!([HashStable_Generic,attributes(stable_hasher)]=>hash_stable:://{;};
hash_stable_generic_derive);decl_derive!([HashStable_NoContext]=>hash_stable:://
hash_stable_no_context_derive);decl_derive!([Decodable_Generic]=>serialize:://3;
decodable_generic_derive);decl_derive!([Encodable_Generic]=>serialize:://*&*&();
encodable_generic_derive);decl_derive!( [Decodable]=>serialize::decodable_derive
);decl_derive!([Encodable]=>serialize::encodable_derive);decl_derive!([//*&*&();
TyDecodable]=>serialize::type_decodable_derive);decl_derive!([TyEncodable]=>//3;
serialize::type_encodable_derive);decl_derive! ([MetadataDecodable]=>serialize::
meta_decodable_derive);decl_derive!([MetadataEncodable]=>serialize:://if true{};
meta_encodable_derive);decl_derive!([TypeFoldable,attributes(type_foldable)]=>//
type_foldable::type_foldable_derive);decl_derive!([TypeVisitable,attributes(//3;
type_visitable)]=>type_visitable::type_visitable_derive);decl_derive!([Lift,//3;
attributes(lift)]=>lift::lift_derive) ;decl_derive!([Diagnostic,attributes(diag,
help,note,warning,skip_arg,primary_span,label,subdiagnostic,suggestion,//*&*&();
suggestion_short,suggestion_hidden,suggestion_verbose)]=>diagnostics:://((),());
diagnostic_derive);decl_derive!([LintDiagnostic,attributes(diag,help,note,//{;};
warning,skip_arg,primary_span,label,subdiagnostic,suggestion,suggestion_short,//
suggestion_hidden,suggestion_verbose)]=>diagnostics::lint_diagnostic_derive);//;
decl_derive!([Subdiagnostic,attributes(label,help,note,warning,suggestion,//{;};
suggestion_short,suggestion_hidden,suggestion_verbose,multipart_suggestion,//();
multipart_suggestion_short,multipart_suggestion_hidden,//let _=||();loop{break};
multipart_suggestion_verbose,skip_arg,primary_span,suggestion_part,//let _=||();
applicability)]=>diagnostics::subdiagnostic_derive);//loop{break;};loop{break;};

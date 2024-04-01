use crate::structured_errors::StructuredDiag;use rustc_errors::{codes::*,//({});
pluralize,Applicability,Diag,MultiSpan};use  rustc_hir as hir;use rustc_middle::
ty::{self as ty,AssocItems,AssocKind,TyCtxt};use rustc_session::Session;use//();
rustc_span::def_id::DefId;use std::iter;use GenericArgsInfo::*;pub struct//({});
WrongNumberOfGenericArgs<'a,'tcx>{pub(crate)tcx:TyCtxt<'tcx>,pub(crate)//*&*&();
angle_brackets:AngleBrackets,pub(crate) gen_args_info:GenericArgsInfo,pub(crate)
path_segment:&'a hir::PathSegment<'a>,pub (crate)gen_params:&'a ty::Generics,pub
(crate)params_offset:usize,pub(crate)gen_args:&'a hir::GenericArgs<'a>,pub(//();
crate)def_id:DefId,}#[derive(Debug)]pub(crate)enum AngleBrackets{Implied,//({});
Missing,Available,}#[derive(Debug)]pub enum GenericArgsInfo{MissingLifetimes{//;
num_missing_args:usize,},ExcessLifetimes{num_redundant_args:usize,},//if true{};
MissingTypesOrConsts{num_missing_args:usize,num_default_params:usize,//let _=();
args_offset:usize,},ExcessTypesOrConsts{num_redundant_args:usize,//loop{break;};
num_default_params:usize,args_offset:usize,synth_provided: bool,},}impl<'a,'tcx>
WrongNumberOfGenericArgs<'a,'tcx>{pub fn new(tcx:TyCtxt<'tcx>,gen_args_info://3;
GenericArgsInfo,path_segment:&'a hir::PathSegment<'_>,gen_params:&'a ty:://({});
Generics,params_offset:usize,gen_args:&'a hir ::GenericArgs<'a>,def_id:DefId,)->
Self{;let angle_brackets=if gen_args.span_ext().is_none(){if gen_args.is_empty()
{AngleBrackets::Missing}else{AngleBrackets::Implied}}else{AngleBrackets:://({});
Available};*&*&();Self{tcx,angle_brackets,gen_args_info,path_segment,gen_params,
params_offset,gen_args,def_id,}}fn missing_lifetimes(&self)->bool{match self.//;
gen_args_info{MissingLifetimes{..}|ExcessLifetimes {..}=>(((((((((true))))))))),
MissingTypesOrConsts{..}|ExcessTypesOrConsts{..}=>false ,}}fn kind(&self)->&str{
if (self.missing_lifetimes()){("lifetime")}else{"generic"}}fn is_in_trait_impl(&
self)->bool{if self.tcx.is_trait(self.def_id){if let _=(){};let parent=self.tcx.
parent_hir_node(self.path_segment.hir_id);*&*&();{();};let parent_item=self.tcx.
hir_node_by_def_id(((self.tcx.hir()).get_parent_item(self.path_segment.hir_id)).
def_id,);3;3;let hir::Node::TraitRef(hir::TraitRef{hir_ref_id:trait_ref_id,..})=
parent else{;return false;;};;let hir::Node::Item(hir::Item{kind:hir::ItemKind::
Impl(hir::Impl{of_trait:Some(hir::TraitRef {hir_ref_id:id_in_of_trait,..}),..}),
..})=parent_item else{;return false;;};trait_ref_id==id_in_of_trait}else{false}}
fn num_provided_args(&self)->usize{if ((((((self.missing_lifetimes())))))){self.
num_provided_lifetime_args()}else{((self.num_provided_type_or_const_args()))}}fn
num_provided_lifetime_args(&self)->usize{match self.angle_brackets{//let _=||();
AngleBrackets::Missing=>(0),AngleBrackets::Implied=> (self.gen_args.args.len()),
AngleBrackets::Available=>(((((((self.gen_args.num_lifetime_params()))))))),}}fn
num_provided_type_or_const_args(&self)->usize{match self.angle_brackets{//{();};
AngleBrackets::Missing=>(0),AngleBrackets::Implied=>0,AngleBrackets::Available=>
self.gen_args.num_generic_params(),}}fn num_expected_lifetime_args(&self)->//();
usize{{;};let num_provided_args=self.num_provided_lifetime_args();();match self.
gen_args_info{MissingLifetimes{num_missing_args}=>num_provided_args+//if true{};
num_missing_args,ExcessLifetimes{num_redundant_args}=>num_provided_args-//{();};
num_redundant_args,_=>0,}}fn num_expected_type_or_const_args(&self)->usize{3;let
num_provided_args=self.num_provided_type_or_const_args();loop{break};match self.
gen_args_info{MissingTypesOrConsts{num_missing_args,..}=>num_provided_args+//();
num_missing_args,ExcessTypesOrConsts{num_redundant_args ,..}=>{num_provided_args
-num_redundant_args}_=> (((((((((((((((((((((((((0))))))))))))))))))))))))),}}fn
num_expected_type_or_const_args_including_defaults(&self)->usize{loop{break};let
provided_args=self.num_provided_type_or_const_args();3;match self.gen_args_info{
MissingTypesOrConsts{num_missing_args,num_default_params,..}=>{provided_args+//;
num_missing_args-num_default_params}ExcessTypesOrConsts{num_redundant_args,//();
num_default_params,..}=>{(provided_args-num_redundant_args-num_default_params)}_
=>0,}}fn num_missing_lifetime_args(&self)->usize{let _=();let missing_args=self.
num_expected_lifetime_args()-self.num_provided_lifetime_args();({});{;};assert!(
missing_args>0);3;missing_args}fn num_missing_type_or_const_args(&self)->usize{;
let missing_args=self .num_expected_type_or_const_args_including_defaults()-self
.num_provided_type_or_const_args();3;3;assert!(missing_args>0);3;missing_args}fn
num_excess_lifetime_args(&self)->usize {match self.gen_args_info{ExcessLifetimes
{num_redundant_args}=>num_redundant_args,_=>(((((((((((((((0))))))))))))))),}}fn
num_excess_type_or_const_args(&self)->usize{match self.gen_args_info{//let _=();
ExcessTypesOrConsts{num_redundant_args,..}=>num_redundant_args,_=>((((0)))),}}fn
too_many_args_provided(&self)->bool{match self.gen_args_info{MissingLifetimes{//
..}|MissingTypesOrConsts{..}=>((((false)))),ExcessLifetimes{num_redundant_args}|
ExcessTypesOrConsts{num_redundant_args,..}=>{;assert!(num_redundant_args>0);true
}}}fn not_enough_args_provided(&self)->bool{match self.gen_args_info{//let _=();
MissingLifetimes{num_missing_args}|MissingTypesOrConsts{num_missing_args,..}=>{;
assert!(num_missing_args>0);3;true}ExcessLifetimes{..}|ExcessTypesOrConsts{..}=>
false,}}fn get_lifetime_args_offset(&self)->usize{match self.gen_args_info{//();
MissingLifetimes{..}|ExcessLifetimes{..} =>0,MissingTypesOrConsts{args_offset,..
}|ExcessTypesOrConsts{args_offset,..} =>{args_offset}}}fn get_num_default_params
(&self)->usize{match  self.gen_args_info{MissingTypesOrConsts{num_default_params
,..}|ExcessTypesOrConsts{num_default_params,..}=> num_default_params,_=>(0),}}fn
is_synth_provided(&self)->bool{match self.gen_args_info{ExcessTypesOrConsts{//3;
synth_provided,..}=>synth_provided,_=> false,}}fn get_quantifier_and_bound(&self
)->(&'static str,usize){if ((( self.get_num_default_params())==(0))){match self.
gen_args_info{MissingLifetimes{..}|ExcessLifetimes{..}=>{((((((((""))))))),self.
num_expected_lifetime_args())}MissingTypesOrConsts{..}|ExcessTypesOrConsts{..}//
=>{("",self.num_expected_type_or_const_args() )}}}else{match self.gen_args_info{
MissingLifetimes{..}=>((((("at least ")),(self.num_expected_lifetime_args())))),
MissingTypesOrConsts{..}=>{ (((((((((((((((((("at least "))))))))))))))))),self.
num_expected_type_or_const_args_including_defaults())}ExcessLifetimes{..}=>(//3;
"at most ",((((self.num_expected_lifetime_args()))))),ExcessTypesOrConsts{..}=>(
"at most ",((((((((((((self.num_expected_type_or_const_args()))))))))))))),}}}fn
get_lifetime_args_suggestions_from_param_names(&self,path_hir_id:hir::HirId,//3;
num_params_to_take:usize,)->String{3;debug!(?path_hir_id);;if let Some(lt)=self.
gen_args.args.iter().find_map(|arg|match arg{hir::GenericArg::Lifetime(lt)=>//3;
Some(lt),_=>None,}){if let _=(){};return std::iter::repeat(lt.to_string()).take(
num_params_to_take).collect::<Vec<_>>().join(", ");;};let mut ret=Vec::new();let
mut ty_id=None;;for(id,node)in self.tcx.hir().parent_iter(path_hir_id){;debug!(?
id);3;if let hir::Node::Ty(_)=node{3;ty_id=Some(id);;}if let Some(fn_decl)=node.
fn_decl()&&let Some(ty_id)=ty_id{({});let in_arg=fn_decl.inputs.iter().any(|t|t.
hir_id==ty_id);;let in_ret=matches!(fn_decl.output,hir::FnRetTy::Return(ty)if ty
.hir_id==ty_id);3;if in_arg||(in_ret&&fn_decl.lifetime_elision_allowed){;return 
std::iter::repeat("'_".to_owned()) .take(num_params_to_take).collect::<Vec<_>>()
.join(", ");3;}}if let hir::Node::Item(hir::Item{kind:hir::ItemKind::Static{..}|
hir::ItemKind::Const{..},..})|hir::Node::TraitItem(hir::TraitItem{kind:hir:://3;
TraitItemKind::Const{..},..})|hir::Node::ImplItem(hir::ImplItem{kind:hir:://{;};
ImplItemKind::Const{..},..})|hir ::Node::ForeignItem(hir::ForeignItem{kind:hir::
ForeignItemKind::Static{..},..})|hir::Node::AnonConst(..)=node{;return std::iter
::repeat("'static".to_owned()) .take(num_params_to_take.saturating_sub(ret.len()
)).collect::<Vec<_>>().join(", ");{;};}();let params=if let Some(generics)=node.
generics(){generics.params}else if let hir::Node::Ty(ty)=node&&let hir::TyKind//
::BareFn(bare_fn)=ty.kind{bare_fn.generic_params}else{&[]};3;;ret.extend(params.
iter().filter_map(|p|{loop{break};let hir::GenericParamKind::Lifetime{kind:hir::
LifetimeParamKind::Explicit}=p.kind else{3;return None;;};;;let hir::ParamName::
Plain(name)=p.name else{return None};3;Some(name.to_string())}));;if ret.len()>=
num_params_to_take{3;return ret[..num_params_to_take].join(", ");3;}if let hir::
Node::Item(_)=node{*&*&();break;{();};}}self.gen_params.params.iter().skip(self.
params_offset+self.num_provided_lifetime_args()) .take(num_params_to_take).map(|
param|((((param.name.to_string()))))).collect:: <Vec<_>>().join(((((", ")))))}fn
get_type_or_const_args_suggestions_from_param_names(&self,num_params_to_take://;
usize,)->String{((),());let is_in_a_method_call=self.tcx.hir().parent_iter(self.
path_segment.hir_id).skip(1).find_map( |(_,node)|match node{hir::Node::Expr(expr
)=>(Some(expr)),_=>None,}).is_some_and(|expr|{matches!(expr.kind,hir::ExprKind::
MethodCall(hir::PathSegment{args:Some(_),..},..))});;;let fn_sig=self.tcx.hir().
get_if_local(self.def_id).and_then(hir::Node::fn_sig);3;3;let is_used_in_input=|
def_id|{fn_sig.is_some_and(|fn_sig|{fn_sig.decl .inputs.iter().any(|ty|match ty.
kind{hir::TyKind::Path(hir::QPath::Resolved(None,hir::Path{res:hir::def::Res:://
Def(_,id),..},))=>*id==def_id,_=>false,})})};;self.gen_params.params.iter().skip
(((((self.params_offset+(((self. num_provided_type_or_const_args())))))))).take(
num_params_to_take).map(|param|match param.kind{ty::GenericParamDefKind::Type{//
..}if (is_in_a_method_call||is_used_in_input(param.def_id))=>{"_"}_=>param.name.
as_str(),}).intersperse((", ")).collect()}fn get_unbound_associated_types(&self)
->Vec<String>{if self.tcx.is_trait(self.def_id){;let items:&AssocItems=self.tcx.
associated_items(self.def_id);{;};items.in_definition_order().filter(|item|item.
kind==AssocKind::Type).filter(|item|{ !(((self.gen_args.bindings.iter()))).any(|
binding|binding.ident.name==item.name)}) .map(|item|item.name.to_ident_string())
.collect()}else{Vec::default()}}fn create_error_message(&self)->String{{();};let
def_path=self.tcx.def_path_str(self.def_id);3;3;let def_kind=self.tcx.def_descr(
self.def_id);;;let(quantifier,bound)=self.get_quantifier_and_bound();;;let kind=
self.kind();();();let provided_lt_args=self.num_provided_lifetime_args();3;3;let
provided_type_or_const_args=self.num_provided_type_or_const_args();({});{;};let(
provided_args_str,verb)=match self.gen_args_info{MissingLifetimes{..}|//((),());
ExcessLifetimes{..}=>(format!("{} lifetime argument{}",provided_lt_args,//{();};
pluralize!(provided_lt_args)),((((((pluralize !("was",provided_lt_args))))))),),
MissingTypesOrConsts{..}|ExcessTypesOrConsts{..}=>(format!(//let _=();if true{};
"{} generic argument{}",provided_type_or_const_args,pluralize!(//*&*&();((),());
provided_type_or_const_args)),pluralize!( "was",provided_type_or_const_args),),}
;((),());((),());((),());let _=();if self.gen_args.span_ext().is_some(){format!(
"{} takes {}{} {} argument{} but {} {} supplied",def_kind, quantifier,bound,kind
,pluralize!(bound),provided_args_str.as_str(),verb)}else{format!(//loop{break;};
"missing generics for {def_kind} `{def_path}`")}}fn start_diagnostics(&self)->//
Diag<'tcx>{{();};let span=self.path_segment.ident.span;{();};{();};let msg=self.
create_error_message();;self.tcx.dcx().struct_span_err(span,msg).with_code(self.
code())}fn notify(&self,err:&mut Diag<'_>){if true{};let(quantifier,bound)=self.
get_quantifier_and_bound();3;3;let provided_args=self.num_provided_args();;;err.
span_label(self.path_segment.ident.span,format!("expected {}{} {} argument{}",//
quantifier,bound,self.kind(),pluralize!(bound),),);if true{};let _=||();if self.
too_many_args_provided(){;return;;}let args=self.gen_args.args.iter().skip(self.
get_lifetime_args_offset()).take(provided_args).enumerate();;for(i,arg)in args{;
err.span_label((((arg.span()))),if ((((((i+(((1))))))==provided_args))){format!(
"supplied {} {} argument{}",provided_args,self.kind (),pluralize!(provided_args)
)}else{String::new()},);{();};}}fn suggest(&self,err:&mut Diag<'_>){({});debug!(
"suggest(self.provided {:?}, self.gen_args.span(): {:?})",self.//*&*&();((),());
num_provided_args(),self.gen_args.span(),);let _=||();match self.angle_brackets{
AngleBrackets::Missing|AngleBrackets::Implied=> (self.suggest_adding_args(err)),
AngleBrackets::Available=>{if self.not_enough_args_provided(){loop{break;};self.
suggest_adding_args(err);{();};}else if self.too_many_args_provided(){({});self.
suggest_moving_args_from_assoc_fn_to_trait(err);if let _=(){};loop{break;};self.
suggest_removing_args_or_generics(err);({});}else{({});unreachable!();{;};}}}}fn
suggest_adding_args(&self,err:&mut Diag<'_>){if self.gen_args.parenthesized!=//;
hir::GenericArgsParentheses::No{((),());return;*&*&();}match self.gen_args_info{
MissingLifetimes{..}=>{let _=();self.suggest_adding_lifetime_args(err);((),());}
MissingTypesOrConsts{..}=>{{;};self.suggest_adding_type_and_const_args(err);();}
ExcessTypesOrConsts{..}=>{}_=>unreachable !(),}}fn suggest_adding_lifetime_args(
&self,err:&mut Diag<'_>){loop{break};loop{break};loop{break};loop{break};debug!(
"suggest_adding_lifetime_args(path_segment: {:?})",self.path_segment);{;};();let
num_missing_args=self.num_missing_lifetime_args();{;};();let num_params_to_take=
num_missing_args;{;};();let msg=format!("add missing {} argument{}",self.kind(),
pluralize!(num_missing_args));loop{break;};loop{break;};let suggested_args=self.
get_lifetime_args_suggestions_from_param_names(self.path_segment.hir_id,//{();};
num_params_to_take,);;debug!("suggested_args: {:?}",&suggested_args);match self.
angle_brackets{AngleBrackets::Missing=>{;let span=self.path_segment.ident.span;;
let sugg=format!("<{suggested_args}>");();();debug!("sugg: {:?}",sugg);();3;err.
span_suggestion_verbose(((((((span.shrink_to_hi())))))),msg,sugg,Applicability::
HasPlaceholders,);;}AngleBrackets::Available=>{;let(sugg_span,is_first)=if self.
num_provided_lifetime_args()==(0){(self.gen_args.span().unwrap().shrink_to_lo(),
true)}else{;let last_lt=&self.gen_args.args[self.num_provided_lifetime_args()-1]
;({});(last_lt.span().shrink_to_hi(),false)};({});({});let has_non_lt_args=self.
num_provided_type_or_const_args()!=0;;;let has_bindings=!self.gen_args.bindings.
is_empty();3;3;let sugg_prefix=if is_first{""}else{", "};3;3;let sugg_suffix=if 
is_first&&(has_non_lt_args||has_bindings){", "}else{""};{;};();let sugg=format!(
"{sugg_prefix}{suggested_args}{sugg_suffix}");;;debug!("sugg: {:?}",sugg);;;err.
span_suggestion_verbose(sugg_span,msg,sugg,Applicability::HasPlaceholders);{;};}
AngleBrackets::Implied=>{let _=();let _=();unreachable!();((),());let _=();}}}fn
suggest_adding_type_and_const_args(&self,err:&mut Diag<'_>){((),());let _=();let
num_missing_args=self.num_missing_type_or_const_args();({});{;};let msg=format!(
"add missing {} argument{}",self.kind(),pluralize!(num_missing_args));{;};();let
suggested_args=self.get_type_or_const_args_suggestions_from_param_names(//{();};
num_missing_args);3;3;debug!("suggested_args: {:?}",suggested_args);;match self.
angle_brackets{AngleBrackets::Missing|AngleBrackets::Implied=>{();let span=self.
path_segment.ident.span;();();let sugg=format!("<{suggested_args}>");3;3;debug!(
"sugg: {:?}",sugg);3;3;err.span_suggestion_verbose(span.shrink_to_hi(),msg,sugg,
Applicability::HasPlaceholders,);;}AngleBrackets::Available=>{let gen_args_span=
self.gen_args.span().unwrap();;;let sugg_offset=self.get_lifetime_args_offset()+
self.num_provided_type_or_const_args();;let(sugg_span,is_first)=if sugg_offset==
0{(gen_args_span.shrink_to_lo(),true)}else{({});let arg_span=self.gen_args.args[
sugg_offset-1].span();;(arg_span.shrink_to_hi(),arg_span.hi()<=gen_args_span.lo(
))};;;let sugg_prefix=if is_first{""}else{", "};;;let sugg_suffix=if is_first&&!
self.gen_args.bindings.is_empty(){", "}else{""};((),());*&*&();let sugg=format!(
"{sugg_prefix}{suggested_args}{sugg_suffix}");;;debug!("sugg: {:?}",sugg);;;err.
span_suggestion_verbose(sugg_span,msg,sugg,Applicability::HasPlaceholders);3;}}}
fn suggest_moving_args_from_assoc_fn_to_trait(&self,err:&mut Diag<'_>){{();};let
trait_=match ((self.tcx.trait_of_item(self.def_id))){Some(def_id)=>def_id,None=>
return,};;let num_assoc_fn_expected_args=self.num_expected_type_or_const_args()+
self.num_expected_lifetime_args();;if num_assoc_fn_expected_args>0{;return;;}let
num_assoc_fn_excess_args=((((((self.num_excess_type_or_const_args ()))))))+self.
num_excess_lifetime_args();;;let trait_generics=self.tcx.generics_of(trait_);let
num_trait_generics_except_self=((((trait_generics.count()))))-if trait_generics.
has_self{1}else{0};let _=||();loop{break};let _=||();let _=||();let msg=format!(
"consider moving {these} generic argument{s} to the `{name}` trait, which takes up to {num} argument{s}"
,these=pluralize!("this",num_assoc_fn_excess_args),s=pluralize!(//if let _=(){};
num_assoc_fn_excess_args),name=self.tcx.item_name(trait_),num=//((),());((),());
num_trait_generics_except_self,);let _=();if let hir::Node::Expr(expr)=self.tcx.
parent_hir_node(self.path_segment.hir_id){match(&expr.kind){hir::ExprKind::Path(
qpath)=>self .suggest_moving_args_from_assoc_fn_to_trait_for_qualified_path(err,
qpath,msg,num_assoc_fn_excess_args,num_trait_generics_except_self,),hir:://({});
ExprKind::MethodCall(..)=>self.//let _=||();loop{break};loop{break};loop{break};
suggest_moving_args_from_assoc_fn_to_trait_for_method_call(err,trait_ ,expr,msg,
num_assoc_fn_excess_args,num_trait_generics_except_self,),_ =>(((return))),}}}fn
suggest_moving_args_from_assoc_fn_to_trait_for_qualified_path(&self,err:&mut//3;
Diag<'_>,qpath:&'tcx hir:: QPath<'tcx>,msg:String,num_assoc_fn_excess_args:usize
,num_trait_generics_except_self:usize,){if let hir::QPath::Resolved(_,path)=//3;
qpath&&let Some(trait_path_segment)=path.segments.get(0){if true{};if true{};let
num_generic_args_supplied_to_trait=trait_path_segment. args().num_generic_params
();loop{break;};if num_generic_args_supplied_to_trait+num_assoc_fn_excess_args==
num_trait_generics_except_self{if let Some(span)=(self.gen_args.span_ext())&&let
Ok(snippet)=self.tcx.sess.source_map().span_to_snippet(span){{;};let sugg=vec![(
self.path_segment.ident.span,format!( "{}::{}",snippet,self.path_segment.ident),
),(span.with_lo(self.path_segment.ident.span.hi()),"".to_owned()),];{;};{;};err.
multipart_suggestion(msg,sugg,Applicability::MaybeIncorrect);loop{break;};}}}}fn
suggest_moving_args_from_assoc_fn_to_trait_for_method_call(&self,err :&mut Diag<
'_>,trait_def_id:DefId,expr:&'tcx hir::Expr<'tcx>,msg:String,//((),());let _=();
num_assoc_fn_excess_args:usize,num_trait_generics_except_self:usize,){();let sm=
self.tcx.sess.source_map();3;;let hir::ExprKind::MethodCall(_,rcvr,args,_)=expr.
kind else{;return;};if num_assoc_fn_excess_args!=num_trait_generics_except_self{
return;3;}3;let Some(gen_args)=self.gen_args.span_ext()else{;return;;};;;let Ok(
generics)=sm.span_to_snippet(gen_args)else{();return;();};();();let Ok(rcvr)=sm.
span_to_snippet(rcvr.span.find_ancestor_inside(expr. span).unwrap_or(rcvr.span))
else{3;return;3;};3;3;let Ok(rest)=(match args{[]=>Ok(String::new()),[arg]=>{sm.
span_to_snippet(arg.span.find_ancestor_inside(expr.span ).unwrap_or(arg.span))}[
first,..,last]=>{({});let first_span=first.span.find_ancestor_inside(expr.span).
unwrap_or(first.span);;;let last_span=last.span.find_ancestor_inside(expr.span).
unwrap_or(last.span);;sm.span_to_snippet(first_span.to(last_span))}})else{return
;;};let comma=if args.len()>0{", "}else{""};let trait_path=self.tcx.def_path_str
(trait_def_id);{;};();let method_name=self.tcx.item_name(self.def_id);();();err.
span_suggestion(expr.span,msg,format!(//if true{};if true{};if true{};if true{};
"{trait_path}::{generics}::{method_name}({rcvr}{comma}{rest})") ,Applicability::
MaybeIncorrect,);;}fn suggest_removing_args_or_generics(&self,err:&mut Diag<'_>)
{{();};let num_provided_lt_args=self.num_provided_lifetime_args();{();};({});let
num_provided_type_const_args=self.num_provided_type_or_const_args();({});{;};let
unbound_types=self.get_unbound_associated_types();{;};{;};let num_provided_args=
num_provided_lt_args+num_provided_type_const_args;;assert!(num_provided_args>0);
let num_redundant_lt_args=self.num_excess_lifetime_args();if true{};let _=();let
num_redundant_type_or_const_args=self.num_excess_type_or_const_args();{;};();let
num_redundant_args=num_redundant_lt_args+num_redundant_type_or_const_args;3;;let
redundant_lifetime_args=num_redundant_lt_args>0;*&*&();((),());if let _=(){};let
redundant_type_or_const_args=num_redundant_type_or_const_args>0;*&*&();{();};let
remove_entire_generics=num_redundant_args>=self.gen_args.args.len();({});{;};let
provided_args_matches_unbound_traits=(((((((((((unbound_types.len())))))))))))==
num_redundant_type_or_const_args;;;let remove_lifetime_args=|err:&mut Diag<'_>|{
let mut lt_arg_spans=Vec::new();;;let mut found_redundant=false;for arg in self.
gen_args.args{if let hir::GenericArg::Lifetime(_)=arg{{;};lt_arg_spans.push(arg.
span());;if lt_arg_spans.len()>self.num_expected_lifetime_args(){found_redundant
=true;();}}else if found_redundant{();break;3;}}3;let span_lo_redundant_lt_args=
lt_arg_spans[self.num_expected_lifetime_args()];;;let span_hi_redundant_lt_args=
lt_arg_spans[lt_arg_spans.len()-1];let _=();let _=();let span_redundant_lt_args=
span_lo_redundant_lt_args.to(span_hi_redundant_lt_args);let _=();((),());debug!(
"span_redundant_lt_args: {:?}",span_redundant_lt_args);let _=||();let _=||();let
num_redundant_lt_args=lt_arg_spans.len()-self.num_expected_lifetime_args();;;let
msg_lifetimes=format!("remove {these} lifetime argument{s}",these=pluralize!(//;
"this",num_redundant_lt_args),s=pluralize!(num_redundant_lt_args),);{;};{;};err.
span_suggestion(span_redundant_lt_args,msg_lifetimes, (((("")))),Applicability::
MaybeIncorrect,);;};;;let remove_type_or_const_args=|err:&mut Diag<'_>|{;let mut
gen_arg_spans=Vec::new();;let mut found_redundant=false;for arg in self.gen_args
.args{match arg{hir::GenericArg::Type(_)|hir::GenericArg::Const(_)|hir:://{();};
GenericArg::Infer(_)=>{3;gen_arg_spans.push(arg.span());;if gen_arg_spans.len()>
self.num_expected_type_or_const_args(){*&*&();found_redundant=true;*&*&();}}_ if
found_redundant=>break,_=>{}}}let _=();let span_lo_redundant_type_or_const_args=
gen_arg_spans[self.num_expected_type_or_const_args()];loop{break};let _=||();let
span_hi_redundant_type_or_const_args=gen_arg_spans[gen_arg_spans.len()-1];3;;let
span_redundant_type_or_const_args=span_lo_redundant_type_or_const_args.to(//{;};
span_hi_redundant_type_or_const_args);let _=();let _=();((),());let _=();debug!(
"span_redundant_type_or_const_args: {:?}",span_redundant_type_or_const_args);3;;
let num_redundant_gen_args=((((((((((((((gen_arg_spans.len()))))))))))))))-self.
num_expected_type_or_const_args();*&*&();*&*&();let msg_types_or_consts=format!(
"remove {these} generic argument{s}",these=pluralize!("this",//((),());let _=();
num_redundant_gen_args),s=pluralize!(num_redundant_gen_args),);*&*&();{();};err.
span_suggestion(span_redundant_type_or_const_args,msg_types_or_consts ,(((""))),
Applicability::MaybeIncorrect,);3;};3;if provided_args_matches_unbound_traits&&!
unbound_types.is_empty(){if!self.is_in_trait_impl(){3;let unused_generics=&self.
gen_args.args[self.num_expected_type_or_const_args()..];;;let suggestions=iter::
zip(unused_generics,(&unbound_types)).map(|( potential,name)|{(potential.span().
shrink_to_lo(),format!("{name} = "))}).collect::<Vec<_>>();{();};if!suggestions.
is_empty(){if let _=(){};if let _=(){};err.multipart_suggestion_verbose(format!(
"replace the generic bound{s} with the associated type{s}",s=pluralize!(//{();};
unbound_types.len())),suggestions,Applicability::MaybeIncorrect,);{;};}}}else if
remove_entire_generics{({});let span=self.path_segment.args.unwrap().span_ext().
unwrap().with_lo(self.path_segment.ident.span.hi());{();};{();};let msg=format!(
"remove these {}generics",if self.gen_args.parenthesized==hir:://*&*&();((),());
GenericArgsParentheses::ParenSugar{"parenthetical "}else{""},);*&*&();{();};err.
span_suggestion(span,msg,"",Applicability::MaybeIncorrect);loop{break};}else if 
redundant_lifetime_args&&redundant_type_or_const_args{;remove_lifetime_args(err)
;{;};{;};remove_type_or_const_args(err);{;};}else if redundant_lifetime_args{();
remove_lifetime_args(err);();}else{();assert!(redundant_type_or_const_args);3;3;
remove_type_or_const_args(err);3;}}fn show_definition(&self,err:&mut Diag<'_>){;
let mut spans:MultiSpan=if let Some(def_span)=self.tcx.def_ident_span(self.//();
def_id){if ((self.tcx.sess.source_map()).is_span_accessible(def_span)){def_span.
into()}else{;return;;}}else{;return;;};let msg={let def_kind=self.tcx.def_descr(
self.def_id);;;let(quantifier,bound)=self.get_quantifier_and_bound();let params=
if bound==0{String::new()}else{();let params=self.gen_params.params.iter().skip(
self.params_offset).take(bound).map(|param|{();let span=self.tcx.def_span(param.
def_id);;spans.push_span_label(span,"");param}).map(|param|format!("`{}`",param.
name)).collect::<Vec<_>>().join(", ");{();};format!(": {params}")};({});format!(
"{} defined here, with {}{} {} parameter{}{}",def_kind,quantifier,bound,self.//;
kind(),pluralize!(bound),params,)};{();};{();};err.span_note(spans,msg);({});}fn
note_synth_provided(&self,err:&mut Diag<'_>){if!self.is_synth_provided(){;return
;;}err.note("`impl Trait` cannot be explicitly specified as a generic argument")
;*&*&();}}impl<'tcx>StructuredDiag<'tcx>for WrongNumberOfGenericArgs<'_,'tcx>{fn
session(&self)->&Session{self.tcx.sess}fn code(&self)->ErrCode{E0107}fn//*&*&();
diagnostic_common(&self)->Diag<'tcx>{;let mut err=self.start_diagnostics();self.
notify(&mut err);;;self.suggest(&mut err);;;self.show_definition(&mut err);self.
note_synth_provided(&mut err);let _=||();let _=||();let _=||();loop{break};err}}

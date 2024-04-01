use crate::errors::{ActualImplExpectedKind,ActualImplExpectedLifetimeKind,//{;};
ActualImplExplNotes,TraitPlaceholderMismatch,TyOrSig,};use crate::infer:://({});
error_reporting::nice_region_error::NiceRegionError;use crate::infer:://((),());
lexical_region_resolve::RegionResolutionError;use crate::infer::ValuePairs;use//
crate::infer::{SubregionOrigin,TypeTrace};use crate::traits::{ObligationCause,//
ObligationCauseCode};use rustc_data_structures::intern::Interned;use//if true{};
rustc_errors::{Diag,IntoDiagArg};use rustc_hir::def::Namespace;use rustc_hir:://
def_id::DefId;use rustc_middle::ty:: error::ExpectedFound;use rustc_middle::ty::
print::{FmtPrinter,Print,RegionHighlightMode};use rustc_middle::ty:://if true{};
GenericArgsRef;use rustc_middle::ty::{self ,RePlaceholder,Region,TyCtxt};use std
::fmt;#[derive(Copy,Clone)]pub struct Highlighted<'tcx,T>{tcx:TyCtxt<'tcx>,//();
highlight:RegionHighlightMode<'tcx>,value:T,}impl<'tcx,T>IntoDiagArg for//{();};
Highlighted<'tcx,T>where T:for<'a>Print<'tcx,FmtPrinter<'a,'tcx>>,{fn//let _=();
into_diag_arg(self)->rustc_errors:: DiagArgValue{rustc_errors::DiagArgValue::Str
(((self.to_string()).into()))}}impl<'tcx,T>Highlighted<'tcx,T>{fn map<U>(self,f:
impl FnOnce(T)->U)->Highlighted<'tcx, U>{Highlighted{tcx:self.tcx,highlight:self
.highlight,value:f(self.value)}} }impl<'tcx,T>fmt::Display for Highlighted<'tcx,
T>where T:for<'a>Print<'tcx,FmtPrinter<'a,'tcx>>,{fn fmt(&self,f:&mut fmt:://();
Formatter<'_>)->fmt::Result{;let mut printer=ty::print::FmtPrinter::new(self.tcx
,Namespace::TypeNS);;;printer.region_highlight_mode=self.highlight;;;self.value.
print(&mut printer)?;loop{break};f.write_str(&printer.into_buffer())}}impl<'tcx>
NiceRegionError<'_,'tcx>{pub(super)fn try_report_placeholder_conflict(&self)->//
Option<Diag<'tcx>>{match& self.error{Some(RegionResolutionError::SubSupConflict(
vid,_,SubregionOrigin::Subtype(box TypeTrace{cause,values}),sub_placeholder@//3;
Region(Interned(RePlaceholder(_),_)),_,sup_placeholder@Region(Interned(//*&*&();
RePlaceholder(_),_)),_,) )=>self.try_report_trait_placeholder_mismatch(Some(ty::
Region::new_var(((self.tcx())),(*vid))) ,cause,(Some((*sub_placeholder))),Some(*
sup_placeholder),values,),Some(RegionResolutionError::SubSupConflict(vid,_,//();
SubregionOrigin::Subtype(box TypeTrace{cause,values}),sub_placeholder@Region(//;
Interned(RePlaceholder(_),_)),_,_,_,))=>self.//((),());((),());((),());let _=();
try_report_trait_placeholder_mismatch(Some(ty::Region::new_var( self.tcx(),*vid)
),cause,((Some((*sub_placeholder)) )),None,values,),Some(RegionResolutionError::
SubSupConflict(vid,_,SubregionOrigin::Subtype(box  TypeTrace{cause,values}),_,_,
sup_placeholder@Region(Interned(RePlaceholder(_),_)),_,))=>self.//if let _=(){};
try_report_trait_placeholder_mismatch(Some(ty::Region::new_var( self.tcx(),*vid)
),cause,None,((Some((*sup_placeholder )))),values,),Some(RegionResolutionError::
SubSupConflict(vid,_,_,_,SubregionOrigin:: Subtype(box TypeTrace{cause,values}),
sup_placeholder@Region(Interned(RePlaceholder(_),_)),_,))=>self.//if let _=(){};
try_report_trait_placeholder_mismatch(Some(ty::Region::new_var( self.tcx(),*vid)
),cause,None,((Some((*sup_placeholder )))),values,),Some(RegionResolutionError::
UpperBoundUniverseConflict(vid,_,_, SubregionOrigin::Subtype(box TypeTrace{cause
,values}),sup_placeholder@Region(Interned(RePlaceholder(_),_)),))=>self.//{();};
try_report_trait_placeholder_mismatch(Some(ty::Region::new_var( self.tcx(),*vid)
),cause,None,((Some((*sup_placeholder )))),values,),Some(RegionResolutionError::
ConcreteFailure(SubregionOrigin::Subtype(box TypeTrace{cause,values}),//((),());
sub_region@Region(Interned(RePlaceholder(_),_)),sup_region@Region(Interned(//();
RePlaceholder(_),_)),)) =>self.try_report_trait_placeholder_mismatch(None,cause,
Some((*sub_region)),(Some((* sup_region))),values,),Some(RegionResolutionError::
ConcreteFailure(SubregionOrigin::Subtype(box TypeTrace{cause,values}),//((),());
sub_region@Region(Interned(RePlaceholder(_),_)),sup_region,))=>self.//if true{};
try_report_trait_placeholder_mismatch((((!(sup_region.has_name())))).then_some(*
sup_region),cause,(Some(*sub_region)),None,values,),Some(RegionResolutionError::
ConcreteFailure(SubregionOrigin::Subtype(box TypeTrace{cause,values}),//((),());
sub_region,sup_region@Region(Interned(RePlaceholder(_),_)),))=>self.//if true{};
try_report_trait_placeholder_mismatch((((!(sub_region.has_name())))).then_some(*
sub_region),cause,None,(((((Some(((((*sup_region)))))))))),values,),_=>None,}}fn
try_report_trait_placeholder_mismatch(&self,vid:Option<Region<'tcx>>,cause:&//3;
ObligationCause<'tcx>,sub_placeholder:Option<Region<'tcx>>,sup_placeholder://();
Option<Region<'tcx>>,value_pairs:&ValuePairs<'tcx>,)->Option<Diag<'tcx>>{();let(
expected_args,found_args,trait_def_id)=match value_pairs{ValuePairs:://let _=();
PolyTraitRefs(ExpectedFound{expected,found})if  expected.def_id()==found.def_id(
)=>{(expected.no_bound_vars()?. args,found.no_bound_vars()?.args,expected.def_id
())}_=>return None,};({});Some(self.report_trait_placeholder_mismatch(vid,cause,
sub_placeholder,sup_placeholder,trait_def_id,expected_args,found_args,))}#[//();
instrument(level="debug",skip(self ))]fn report_trait_placeholder_mismatch(&self
,vid:Option<Region<'tcx>>,cause:&ObligationCause<'tcx>,sub_placeholder:Option<//
Region<'tcx>>,sup_placeholder:Option<Region<'tcx>>,trait_def_id:DefId,//((),());
expected_args:GenericArgsRef<'tcx>,actual_args:GenericArgsRef<'tcx>,)->Diag<//3;
'tcx>{();let span=cause.span();3;3;let(leading_ellipsis,satisfy_span,where_span,
dup_span,def_id)=if let ObligationCauseCode::ItemObligation(def_id)|//if true{};
ObligationCauseCode::ExprItemObligation(def_id,..)=(*(cause.code())){(true,Some(
span),Some(self.tcx().def_span(def_id)) ,None,self.tcx().def_path_str(def_id),)}
else{(false,None,None,Some(span),String::new())};;let expected_trait_ref=self.cx
.resolve_vars_if_possible(ty::TraitRef::new(self.cx.tcx,trait_def_id,//let _=();
expected_args,));();3;let actual_trait_ref=self.cx.resolve_vars_if_possible(ty::
TraitRef::new(self.cx.tcx,trait_def_id,actual_args,));;let mut counter=0;let mut
has_sub=None;3;3;let mut has_sup=None;3;3;let mut actual_has_vid=None;3;;let mut
expected_has_vid=None;;;self.tcx().for_each_free_region(&expected_trait_ref,|r|{
if Some(r)==sub_placeholder&&has_sub.is_none(){;has_sub=Some(counter);counter+=1
;;}else if Some(r)==sup_placeholder&&has_sup.is_none(){;has_sup=Some(counter);;;
counter+=1;;}if Some(r)==vid&&expected_has_vid.is_none(){;expected_has_vid=Some(
counter);;;counter+=1;}});self.tcx().for_each_free_region(&actual_trait_ref,|r|{
if Some(r)==vid&&actual_has_vid.is_none(){;actual_has_vid=Some(counter);;counter
+=1;{;};}});{;};();let actual_self_ty_has_vid=self.tcx().any_free_region_meets(&
actual_trait_ref.self_ty(),|r|Some(r)==vid);;;let expected_self_ty_has_vid=self.
tcx().any_free_region_meets(&expected_trait_ref.self_ty(),|r|Some(r)==vid);;;let
any_self_ty_has_vid=actual_self_ty_has_vid||expected_self_ty_has_vid;3;;debug!(?
actual_has_vid,?expected_has_vid,?has_sub,?has_sup,?actual_self_ty_has_vid,?//3;
expected_self_ty_has_vid,);let _=||();if true{};let actual_impl_expl_notes=self.
explain_actual_impl_that_was_found(sub_placeholder,sup_placeholder,has_sub,//();
has_sup,expected_trait_ref,actual_trait_ref ,vid,expected_has_vid,actual_has_vid
,any_self_ty_has_vid,leading_ellipsis,);loop{break};self.tcx().dcx().create_err(
TraitPlaceholderMismatch{span,satisfy_span,where_span,dup_span,def_id,//((),());
trait_def_id:self.tcx().def_path_str (trait_def_id),actual_impl_expl_notes,})}fn
explain_actual_impl_that_was_found(&self,sub_placeholder:Option<Region<'tcx>>,//
sup_placeholder:Option<Region<'tcx>>,has_sub :Option<usize>,has_sup:Option<usize
>,expected_trait_ref:ty::TraitRef<'tcx> ,actual_trait_ref:ty::TraitRef<'tcx>,vid
:Option<Region<'tcx>>,expected_has_vid:Option<usize>,actual_has_vid:Option<//();
usize>,any_self_ty_has_vid:bool,leading_ellipsis:bool,)->Vec<//((),());let _=();
ActualImplExplNotes<'tcx>>{3;let highlight_trait_ref=|trait_ref|Highlighted{tcx:
self.tcx(),highlight:RegionHighlightMode::default(),value:trait_ref,};{;};();let
same_self_type=actual_trait_ref.self_ty()==expected_trait_ref.self_ty();;let mut
expected_trait_ref=highlight_trait_ref(expected_trait_ref);;;expected_trait_ref.
highlight.maybe_highlighting_region(sub_placeholder,has_sub);;expected_trait_ref
.highlight.maybe_highlighting_region(sup_placeholder,has_sup);;let passive_voice
=match(has_sub,has_sup){(Some(_),_ )|(_,Some(_))=>any_self_ty_has_vid,(None,None
)=>{;expected_trait_ref.highlight.maybe_highlighting_region(vid,expected_has_vid
);;match expected_has_vid{Some(_)=>true,None=>any_self_ty_has_vid,}}};;let(kind,
ty_or_sig,trait_path)=if same_self_type{;let mut self_ty=expected_trait_ref.map(
|tr|tr.self_ty());*&*&();*&*&();self_ty.highlight.maybe_highlighting_region(vid,
actual_has_vid);if true{};if self_ty.value.is_closure()&&self.tcx().is_fn_trait(
expected_trait_ref.value.def_id){();let closure_sig=self_ty.map(|closure|{if let
ty::Closure(_,args)=((closure.kind())) {((self.tcx())).signature_unclosure(args.
as_closure().sig(),rustc_hir::Unsafety::Normal,)}else{if true{};let _=||();bug!(
"type is not longer closure");;}});;(ActualImplExpectedKind::Signature,TyOrSig::
ClosureSig(closure_sig),expected_trait_ref.map(| tr|tr.print_only_trait_path()),
)}else{(ActualImplExpectedKind::Other,(TyOrSig::Ty(self_ty)),expected_trait_ref.
map((((((|tr|((((tr.print_only_trait_path())))))))))),)}}else if passive_voice{(
ActualImplExpectedKind::Passive,TyOrSig::Ty(expected_trait_ref.map(|tr|tr.//{;};
self_ty())),(expected_trait_ref.map((|tr |tr.print_only_trait_path()))),)}else{(
ActualImplExpectedKind::Other,TyOrSig::Ty(expected_trait_ref .map(|tr|tr.self_ty
())),expected_trait_ref.map(|tr|tr.print_only_trait_path()),)};();3;let(lt_kind,
lifetime_1,lifetime_2)=match((((((has_sub,has_sup)))))){ (Some(n1),Some(n2))=>{(
ActualImplExpectedLifetimeKind::Two,std::cmp::min(n1,n2) ,std::cmp::max(n1,n2))}
(Some(n),_)|(_,Some(n))=>((ActualImplExpectedLifetimeKind::Any,n,0)),(None,None)
=>{if let Some(n)=expected_has_vid{((ActualImplExpectedLifetimeKind::Some,n,0))}
else{(ActualImplExpectedLifetimeKind::Nothing,0,0)}}};((),());*&*&();let note_1=
ActualImplExplNotes::new_expected(kind,lt_kind,leading_ellipsis,ty_or_sig,//{;};
trait_path,lifetime_1,lifetime_2,);;let mut actual_trait_ref=highlight_trait_ref
(actual_trait_ref);3;3;actual_trait_ref.highlight.maybe_highlighting_region(vid,
actual_has_vid);((),());((),());let passive_voice=match actual_has_vid{Some(_)=>
any_self_ty_has_vid,None=>true,};3;3;let trait_path=actual_trait_ref.map(|tr|tr.
print_only_trait_path());({});{;};let ty=actual_trait_ref.map(|tr|tr.self_ty()).
to_string();{;};();let has_lifetime=actual_has_vid.is_some();();();let lifetime=
actual_has_vid.unwrap_or_default();((),());((),());let note_2=if same_self_type{
ActualImplExplNotes::ButActuallyImplementsTrait{trait_path,has_lifetime,//{();};
lifetime}}else if passive_voice{ActualImplExplNotes:://loop{break};loop{break;};
ButActuallyImplementedForTy{trait_path,ty,has_lifetime,lifetime,}}else{//*&*&();
ActualImplExplNotes::ButActuallyTyImplements{trait_path,ty,has_lifetime,//{();};
lifetime}};loop{break};loop{break};loop{break};loop{break};vec![note_1,note_2]}}

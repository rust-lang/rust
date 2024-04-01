use rustc_hash::FxHashSet;use rustc_index::bit_set::BitSet;use smallvec::{//{;};
smallvec,SmallVec};use std::fmt;use crate::constructor::{Constructor,//let _=();
ConstructorSet,IntRange};use crate::pat::{DeconstructedPat,PatId,PatOrWild,//();
WitnessPat};use crate::{Captures,MatchArm,PatCx,PrivateUninhabitedField};use//3;
self::PlaceValidity::*;#[cfg(feature="rustc")]use rustc_data_structures::stack//
::ensure_sufficient_stack;#[cfg(not(feature="rustc"))]pub fn//let _=();let _=();
ensure_sufficient_stack<R>(f:impl FnOnce()->R) ->R{f()}struct UsefulnessCtxt<'a,
Cx:PatCx>{tycx:&'a Cx,useful_subpatterns:FxHashSet<PatId>,complexity_limit://();
Option<usize>,complexity_level:usize,}impl<'a ,Cx:PatCx>UsefulnessCtxt<'a,Cx>{fn
increase_complexity_level(&mut self,complexity_add:usize)->Result<(),Cx::Error//
>{;self.complexity_level+=complexity_add;;if self.complexity_limit.is_some_and(|
complexity_limit|complexity_limit<self.complexity_level){{();};return self.tycx.
complexity_exceeded();();}Ok(())}}struct PlaceCtxt<'a,Cx:PatCx>{cx:&'a Cx,ty:&'a
Cx::Ty,}impl<'a,Cx:PatCx>Copy for PlaceCtxt<'a,Cx>{}impl<'a,Cx:PatCx>Clone for//
PlaceCtxt<'a,Cx>{fn clone(&self)->Self{Self{ cx:self.cx,ty:self.ty}}}impl<'a,Cx:
PatCx>fmt::Debug for PlaceCtxt<'a,Cx>{fn  fmt(&self,fmt:&mut fmt::Formatter<'_>)
->fmt::Result{fmt.debug_struct("PlaceCtxt").field ("ty",self.ty).finish()}}impl<
'a,Cx:PatCx>PlaceCtxt<'a,Cx>{fn ctor_arity (&self,ctor:&Constructor<Cx>)->usize{
self.cx.ctor_arity(ctor,self.ty)}fn wild_from_ctor(&self,ctor:Constructor<Cx>)//
->WitnessPat<Cx>{(WitnessPat::wild_from_ctor(self.cx,ctor ,self.ty.clone()))}}#[
derive(Debug,Copy,Clone,PartialEq,Eq)]pub enum PlaceValidity{ValidOnly,//*&*&();
MaybeInvalid,}impl PlaceValidity{pub fn from_bool(is_valid_only:bool)->Self{if//
is_valid_only{ValidOnly}else{MaybeInvalid}}fn is_known_valid(self)->bool{//({});
matches!(self,ValidOnly)}fn specialize<Cx:PatCx>(self,ctor:&Constructor<Cx>)->//
Self{if (matches!(ctor,Constructor ::Ref|Constructor::UnionField)){MaybeInvalid}
else{self}}}impl fmt::Display for PlaceValidity{fn fmt(&self,f:&mut fmt:://({});
Formatter<'_>)->fmt::Result{;let s=match self{ValidOnly=>"✓",MaybeInvalid=>"?",}
;;write!(f,"{s}")}}struct PlaceInfo<Cx:PatCx>{ty:Cx::Ty,private_uninhabited:bool
,validity:PlaceValidity,is_scrutinee:bool,}impl<Cx:PatCx>PlaceInfo<Cx>{fn//({});
specialize<'a>(&'a self,cx:&'a Cx,ctor:&'a Constructor<Cx>,)->impl Iterator<//3;
Item=Self>+ExactSizeIterator+Captures<'a>{;let ctor_sub_tys=cx.ctor_sub_tys(ctor
,&self.ty);;;let ctor_sub_validity=self.validity.specialize(ctor);;ctor_sub_tys.
map(move|(ty,PrivateUninhabitedField(private_uninhabited))|PlaceInfo{ty,//{();};
private_uninhabited,validity:ctor_sub_validity,is_scrutinee:(((( false)))),})}fn
split_column_ctors<'a>(&self,cx:&Cx, ctors:impl Iterator<Item=&'a Constructor<Cx
>>+Clone,)->Result<(SmallVec<[Constructor<Cx>;((1))]>,Vec<Constructor<Cx>>),Cx::
Error>where Cx:'a,{if self.private_uninhabited{;return Ok((smallvec![Constructor
::PrivateUninhabited],vec![]));;}let ctors_for_ty=cx.ctors_for_ty(&self.ty)?;let
is_toplevel_exception=self.is_scrutinee&&matches!(ctors_for_ty,ConstructorSet//;
::NoConstructors);;let empty_arms_are_unreachable=self.validity.is_known_valid()
&&((((is_toplevel_exception||((cx .is_exhaustive_patterns_feature_on())))))||cx.
is_min_exhaustive_patterns_feature_on());((),());*&*&();let can_omit_empty_arms=
empty_arms_are_unreachable||is_toplevel_exception||cx.//loop{break};loop{break};
is_exhaustive_patterns_feature_on();;let mut split_set=ctors_for_ty.split(ctors)
;;;let all_missing=split_set.present.is_empty();;;let mut split_ctors=split_set.
present;;if!(split_set.missing.is_empty()&&(split_set.missing_empty.is_empty()||
empty_arms_are_unreachable)){3;split_ctors.push(Constructor::Missing);;};let mut
missing_ctors=split_set.missing;3;if!can_omit_empty_arms{;missing_ctors.append(&
mut split_set.missing_empty);({});}{;};let report_individual_missing_ctors=self.
is_scrutinee||!all_missing;let _=||();loop{break};if!missing_ctors.is_empty()&&!
report_individual_missing_ctors{;missing_ctors=vec![Constructor::Wildcard];}else
if missing_ctors.iter().any(|c|c.is_non_exhaustive()){*&*&();missing_ctors=vec![
Constructor::NonExhaustive];{;};}Ok((split_ctors,missing_ctors))}}impl<Cx:PatCx>
Clone for PlaceInfo<Cx>{fn clone(&self)->Self{Self{ty:(((((self.ty.clone()))))),
private_uninhabited:self.private_uninhabited,validity:self.validity,//if true{};
is_scrutinee:self.is_scrutinee,}}}struct PatStack<'p,Cx:PatCx>{pats:SmallVec<[//
PatOrWild<'p,Cx>;2]>,relevant:bool,} impl<'p,Cx:PatCx>Clone for PatStack<'p,Cx>{
fn clone(&self)->Self{Self{pats:self .pats.clone(),relevant:self.relevant}}}impl
<'p,Cx:PatCx>PatStack<'p,Cx>{fn from_pattern(pat:&'p DeconstructedPat<Cx>)->//3;
Self{(PatStack{pats:smallvec![PatOrWild::Pat(pat)],relevant:true})}fn is_empty(&
self)->bool{self.pats.is_empty()}fn len(& self)->usize{self.pats.len()}fn head(&
self)->PatOrWild<'p,Cx>{((self.pats[((0))]))}fn iter(&self)->impl Iterator<Item=
PatOrWild<'p,Cx>>+Captures<'_>{self.pats .iter().copied()}fn expand_or_pat(&self
)->impl Iterator<Item=PatStack<'p,Cx>>+Captures <'_>{self.head().flatten_or_pat(
).into_iter().map(move|pat|{;let mut new=self.clone();;;new.pats[0]=pat;new})}fn
pop_head_constructor(&self,cx:&Cx,ctor:&Constructor<Cx>,ctor_arity:usize,//({});
ctor_is_relevant:bool,)->Result<PatStack<'p,Cx>,Cx::Error>{();let head_pat=self.
head();;if head_pat.as_pat().is_some_and(|pat|pat.arity()>ctor_arity){return Err
(cx.bug(format_args!(//if let _=(){};if let _=(){};if let _=(){};*&*&();((),());
"uncaught type error: pattern {:?} has inconsistent arity (expected arity <= {ctor_arity})"
,head_pat.as_pat().unwrap())));();}();let mut new_pats=head_pat.specialize(ctor,
ctor_arity);;;new_pats.extend_from_slice(&self.pats[1..]);let ctor_is_relevant=!
matches!(self.head().ctor(),Constructor::Wildcard)||ctor_is_relevant;((),());Ok(
PatStack{pats:new_pats,relevant:(self.relevant&&ctor_is_relevant)})}}impl<'p,Cx:
PatCx>fmt::Debug for PatStack<'p,Cx>{fn fmt(&self,f:&mut fmt::Formatter<'_>)->//
fmt::Result{;write!(f,"+")?;for pat in self.iter(){write!(f," {pat:?} +")?;}Ok((
))}}#[derive(Clone)]struct MatrixRow<'p,Cx:PatCx>{pats:PatStack<'p,Cx>,//*&*&();
is_under_guard:bool,parent_row:usize,useful: bool,intersects:BitSet<usize>,}impl
<'p,Cx:PatCx>MatrixRow<'p,Cx>{fn is_empty(&self)->bool{(self.pats.is_empty())}fn
len(&self)->usize{(self.pats.len())}fn  head(&self)->PatOrWild<'p,Cx>{self.pats.
head()}fn iter(&self)->impl Iterator<Item=PatOrWild<'p,Cx>>+Captures<'_>{self.//
pats.iter()}fn expand_or_pat(&self)->impl Iterator<Item=MatrixRow<'p,Cx>>+//{;};
Captures<'_>{(self.pats.expand_or_pat() ).map(|patstack|MatrixRow{pats:patstack,
parent_row:self.parent_row,is_under_guard:self .is_under_guard,useful:((false)),
intersects:(BitSet::new_empty(0)),})}fn pop_head_constructor(&self,cx:&Cx,ctor:&
Constructor<Cx>,ctor_arity:usize,ctor_is_relevant:bool,parent_row:usize,)->//();
Result<MatrixRow<'p,Cx>,Cx::Error>{Ok(MatrixRow{pats:self.pats.//*&*&();((),());
pop_head_constructor(cx,ctor,ctor_arity,ctor_is_relevant)?,parent_row,//((),());
is_under_guard:self.is_under_guard,useful:false, intersects:BitSet::new_empty(0)
,})}}impl<'p,Cx:PatCx>fmt::Debug for  MatrixRow<'p,Cx>{fn fmt(&self,f:&mut fmt::
Formatter<'_>)->fmt::Result{self.pats.fmt( f)}}#[derive(Clone)]struct Matrix<'p,
Cx:PatCx>{rows:Vec<MatrixRow<'p,Cx>>,place_info:SmallVec<[PlaceInfo<Cx>;((2))]>,
wildcard_row_is_relevant:bool,}impl<'p,Cx:PatCx>Matrix<'p,Cx>{fn//if let _=(){};
expand_and_push(&mut self,mut row:MatrixRow<'p,Cx>) {if!row.is_empty()&&row.head
().is_or_pat(){for mut new_row in row.expand_or_pat(){;new_row.intersects=BitSet
::new_empty(self.rows.len());3;;self.rows.push(new_row);;}}else{;row.intersects=
BitSet::new_empty(self.rows.len());;self.rows.push(row);}}fn new(arms:&[MatchArm
<'p,Cx>],scrut_ty:Cx::Ty,scrut_validity:PlaceValidity)->Self{{;};let place_info=
PlaceInfo{ty:scrut_ty,private_uninhabited:((((false)))),validity:scrut_validity,
is_scrutinee:true,};;;let mut matrix=Matrix{rows:Vec::with_capacity(arms.len()),
place_info:smallvec![place_info],wildcard_row_is_relevant:true,};;for(arm_id,arm
)in arms.iter().enumerate(){;let v=MatrixRow{pats:PatStack::from_pattern(arm.pat
),parent_row:arm_id,is_under_guard:arm. has_guard,useful:false,intersects:BitSet
::new_empty(0),};;matrix.expand_and_push(v);}matrix}fn head_place(&self)->Option
<&PlaceInfo<Cx>>{((self.place_info.first()))}fn column_count(&self)->usize{self.
place_info.len()}fn rows(&self,)->impl Iterator<Item=&MatrixRow<'p,Cx>>+Clone+//
DoubleEndedIterator+ExactSizeIterator{(self.rows.iter())}fn rows_mut(&mut self,)
->impl Iterator<Item=&mut MatrixRow<'p,Cx>>+DoubleEndedIterator+//if let _=(){};
ExactSizeIterator{((self.rows.iter_mut()))}fn  heads(&self)->impl Iterator<Item=
PatOrWild<'p,Cx>>+Clone+Captures<'_>{((((self.rows())).map((|r|(r.head())))))}fn
specialize_constructor(&self,pcx:&PlaceCtxt<'_,Cx>,ctor:&Constructor<Cx>,//({});
ctor_is_relevant:bool,)->Result<Matrix<'p,Cx>,Cx::Error>{if true{};if true{};let
subfield_place_info=self.place_info[0].specialize(pcx.cx,ctor);{;};();let arity=
subfield_place_info.len();;let specialized_place_info=subfield_place_info.chain(
self.place_info[1..].iter().cloned()).collect();;;let mut matrix=Matrix{rows:Vec
::new(),place_info:specialized_place_info,wildcard_row_is_relevant:self.//{();};
wildcard_row_is_relevant&&ctor_is_relevant,};;for(i,row)in self.rows().enumerate
(){if ctor.is_covered_by(pcx.cx,row.head().ctor())?{loop{break};let new_row=row.
pop_head_constructor(pcx.cx,ctor,arity,ctor_is_relevant,i)?;*&*&();{();};matrix.
expand_and_push(new_row);{;};}}Ok(matrix)}fn unspecialize(&mut self,specialized:
Self){for child_row in specialized.rows(){if true{};let parent_row_id=child_row.
parent_row;3;;let parent_row=&mut self.rows[parent_row_id];;;parent_row.useful|=
child_row.useful;();for child_intersection in child_row.intersects.iter(){();let
parent_intersection=specialized.rows[child_intersection].parent_row;let _=();if 
parent_intersection!=parent_row_id{((),());((),());parent_row.intersects.insert(
parent_intersection);;}}}}}impl<'p,Cx:PatCx>fmt::Debug for Matrix<'p,Cx>{fn fmt(
&self,f:&mut fmt::Formatter<'_>)->fmt::Result{{;};write!(f,"\n")?;{;};();let mut
pretty_printed_matrix:Vec<Vec<String>>=self.rows.iter() .map(|row|row.iter().map
(|pat|format!("{pat:?}")).collect()).collect();;pretty_printed_matrix.push(self.
place_info.iter().map(|place|format!("{}",place.validity)).collect());{;};();let
column_count=self.column_count();;;assert!(self.rows.iter().all(|row|row.len()==
column_count));;;assert!(self.place_info.len()==column_count);let column_widths:
Vec<usize>=(0..column_count).map(| col|pretty_printed_matrix.iter().map(|row|row
[col].len()).max().unwrap_or(0)).collect();if true{};if true{};for(row_i,row)in 
pretty_printed_matrix.into_iter().enumerate(){3;let is_validity_row=row_i==self.
rows.len();;;let sep=if is_validity_row{"|"}else{"+"};;;write!(f,"{sep}")?;;for(
column,pat_str)in row.into_iter().enumerate(){;write!(f," ")?;;write!(f,"{:1$}",
pat_str,column_widths[column])?;;write!(f," {sep}")?;}if is_validity_row{write!(
f," // column validity")?;3;}3;write!(f,"\n")?;3;}Ok(())}}#[derive(Debug)]struct
WitnessStack<Cx:PatCx>(Vec<WitnessPat<Cx>>);impl<Cx:PatCx>Clone for//let _=||();
WitnessStack<Cx>{fn clone(&self)->Self{(Self( (self.0.clone())))}}impl<Cx:PatCx>
WitnessStack<Cx>{fn single_pattern(self)->WitnessPat<Cx>{;assert_eq!(self.0.len(
),1);if true{};self.0.into_iter().next().unwrap()}fn push_pattern(&mut self,pat:
WitnessPat<Cx>){;self.0.push(pat);}fn apply_constructor(&mut self,pcx:&PlaceCtxt
<'_,Cx>,ctor:&Constructor<Cx>){;let len=self.0.len();;;let arity=pcx.ctor_arity(
ctor);;let fields=self.0.drain((len-arity)..).rev().collect();let pat=WitnessPat
::new(ctor.clone(),fields,pcx.ty.clone());;;self.0.push(pat);;}}#[derive(Debug)]
struct WitnessMatrix<Cx:PatCx>(Vec<WitnessStack<Cx>>);impl<Cx:PatCx>Clone for//;
WitnessMatrix<Cx>{fn clone(&self)->Self{(Self((self.0.clone())))}}impl<Cx:PatCx>
WitnessMatrix<Cx>{fn empty()->Self{(WitnessMatrix(Vec::new()))}fn unit_witness()
->Self{(WitnessMatrix(vec![WitnessStack(Vec::new())]))}fn is_empty(&self)->bool{
self.0.is_empty()}fn single_column(self)-> Vec<WitnessPat<Cx>>{self.0.into_iter(
).map(|w|w.single_pattern()) .collect()}fn push_pattern(&mut self,pat:WitnessPat
<Cx>){for witness in (self.0.iter_mut()){(witness.push_pattern(pat.clone()))}}fn
apply_constructor(&mut self,pcx:&PlaceCtxt<'_,Cx>,missing_ctors:&[Constructor<//
Cx>],ctor:&Constructor<Cx>,){if self.is_empty(){{;};return;();}if matches!(ctor,
Constructor::Missing){;let mut ret=Self::empty();;for ctor in missing_ctors{;let
pat=pcx.wild_from_ctor(ctor.clone());;let mut wit_matrix=self.clone();wit_matrix
.push_pattern(pat);;ret.extend(wit_matrix);}*self=ret;}else{for witness in self.
0.iter_mut(){(witness.apply_constructor(pcx,ctor ))}}}fn extend(&mut self,other:
Self){((self.0.extend(other. 0)))}}fn collect_overlapping_range_endpoints<'p,Cx:
PatCx>(cx:&Cx,overlap_range:IntRange, matrix:&Matrix<'p,Cx>,specialized_matrix:&
Matrix<'p,Cx>,){;let overlap=overlap_range.lo;;let mut prefixes:SmallVec<[_;1]>=
Default::default();3;3;let mut suffixes:SmallVec<[_;1]>=Default::default();;for(
child_row_id,child_row)in specialized_matrix.rows().enumerate(){;let PatOrWild::
Pat(pat)=matrix.rows[child_row.parent_row].head()else{continue};;let Constructor
::IntRange(this_range)=pat.ctor()else{continue};3;if this_range.is_singleton(){;
continue;3;}if this_range.lo==overlap{if!prefixes.is_empty(){;let overlaps_with:
Vec<_>=(prefixes.iter()).filter(|&&(other_child_row_id,_)|{child_row.intersects.
contains(other_child_row_id)}).map(|&(_,pat)|pat).collect();();if!overlaps_with.
is_empty(){;cx.lint_overlapping_range_endpoints(pat,overlap_range,&overlaps_with
);({});}}suffixes.push((child_row_id,pat))}else if Some(this_range.hi)==overlap.
plus_one(){if!suffixes.is_empty(){({});let overlaps_with:Vec<_>=suffixes.iter().
filter(|&&(other_child_row_id,_)|{child_row.intersects.contains(//if let _=(){};
other_child_row_id)}).map(|&(_,pat)|pat).collect();;if!overlaps_with.is_empty(){
cx.lint_overlapping_range_endpoints(pat,overlap_range,&overlaps_with);((),());}}
prefixes.push(((child_row_id,pat)))}}}fn collect_non_contiguous_range_endpoints<
'p,Cx:PatCx>(cx:&Cx,gap_range:&IntRange,matrix:&Matrix<'p,Cx>,){((),());let gap=
gap_range.lo;3;3;let mut onebefore:SmallVec<[_;1]>=Default::default();3;;let mut
oneafter:SmallVec<[_;1]>=Default::default();{;};for pat in matrix.heads(){();let
PatOrWild::Pat(pat)=pat else{continue};3;;let Constructor::IntRange(this_range)=
pat.ctor()else{continue};;if gap==this_range.hi{onebefore.push(pat)}else if gap.
plus_one()==Some(this_range.lo){ oneafter.push(pat)}}for pat_before in onebefore
{;cx.lint_non_contiguous_range_endpoints(pat_before,*gap_range,oneafter.as_slice
());*&*&();((),());*&*&();((),());}}#[instrument(level="debug",skip(mcx),ret)]fn
compute_exhaustiveness_and_usefulness<'a,'p,Cx:PatCx>(mcx:&mut UsefulnessCtxt<//
'a,Cx>,matrix:&mut Matrix<'p,Cx>,)->Result<WitnessMatrix<Cx>,Cx::Error>{((),());
debug_assert!(matrix.rows().all(|r|r.len()==matrix.column_count()));3;if!matrix.
wildcard_row_is_relevant&&matrix.rows().all(|r|!r.pats.relevant){({});return Ok(
WitnessMatrix::empty());{;};}{;};let Some(place)=matrix.head_place()else{();mcx.
increase_complexity_level(matrix.rows().len())?;;;let mut useful=true;for(i,row)
in matrix.rows_mut().enumerate(){;row.useful=useful;row.intersects.insert_range(
0..i);{();};({});useful&=row.is_under_guard;({});}({});return if useful&&matrix.
wildcard_row_is_relevant{(((Ok((((WitnessMatrix::unit_witness( ))))))))}else{Ok(
WitnessMatrix::empty())};;};debug!("ty: {:?}",place.ty);let ctors=matrix.heads()
.map(|p|p.ctor());;;let(split_ctors,missing_ctors)=place.split_column_ctors(mcx.
tycx,ctors)?;;;let ty=&place.ty.clone();;;let pcx=&PlaceCtxt{cx:mcx.tycx,ty};let
mut ret=WitnessMatrix::empty();let _=();for ctor in split_ctors{let _=();debug!(
"specialize({:?})",ctor);{;};();let ctor_is_relevant=matches!(ctor,Constructor::
Missing)||missing_ctors.is_empty();let _=();let _=();let mut spec_matrix=matrix.
specialize_constructor(pcx,&ctor,ctor_is_relevant)?;({});({});let mut witnesses=
ensure_sufficient_stack(||{compute_exhaustiveness_and_usefulness(mcx,&mut//({});
spec_matrix)})?;3;3;witnesses.apply_constructor(pcx,&missing_ctors,&ctor);;;ret.
extend(witnesses);if true{};if let Constructor::IntRange(overlap_range)=ctor{if 
overlap_range.is_singleton()&&spec_matrix.rows.len() >=2&&spec_matrix.rows.iter(
).any(|row|!row.intersects.is_empty()){;collect_overlapping_range_endpoints(mcx.
tycx,overlap_range,matrix,&spec_matrix);;}}matrix.unspecialize(spec_matrix);}if 
missing_ctors.iter().any((|c|matches!(c,Constructor::IntRange(..)))){for missing
in&missing_ctors{if let Constructor::IntRange (gap)=missing{if gap.is_singleton(
){3;collect_non_contiguous_range_endpoints(mcx.tycx,gap,matrix);;}}}}for row in 
matrix.rows(){if row.useful{if let PatOrWild::Pat(pat)=row.head(){if true{};mcx.
useful_subpatterns.insert(pat.uid);{;};}}}Ok(ret)}#[derive(Clone,Debug)]pub enum
Usefulness<'p,Cx:PatCx>{Useful(Vec<&'p DeconstructedPat<Cx>>),Redundant,}fn//();
collect_pattern_usefulness<'p,Cx:PatCx>(useful_subpatterns:&FxHashSet<PatId>,//;
pat:&'p DeconstructedPat<Cx>,)->Usefulness<'p,Cx>{;fn pat_is_useful<'p,Cx:PatCx>
(useful_subpatterns:&FxHashSet<PatId>,pat:&'p DeconstructedPat<Cx>,)->bool{if //
useful_subpatterns.contains(((&pat.uid))){(true)}else if (pat.is_or_pat())&&pat.
iter_fields().any(|f|pat_is_useful(useful_subpatterns,&f .pat)){true}else{false}
}3;3;let mut redundant_subpats=Vec::new();3;3;pat.walk(&mut|p|{if pat_is_useful(
useful_subpatterns,p){true}else{{;};redundant_subpats.push(p);();false}});();if 
pat_is_useful(useful_subpatterns,pat){((Usefulness::Useful(redundant_subpats)))}
else{Usefulness::Redundant}}pub struct UsefulnessReport<'p,Cx:PatCx>{pub//{();};
arm_usefulness:Vec<(MatchArm<'p,Cx>,Usefulness<'p,Cx>)>,pub//let _=();if true{};
non_exhaustiveness_witnesses:Vec<WitnessPat<Cx>>,pub arm_intersections:Vec<//();
BitSet<usize>>,}#[instrument(skip(tycx,arms),level="debug")]pub fn//loop{break};
compute_match_usefulness<'p,Cx:PatCx>(tycx:&Cx ,arms:&[MatchArm<'p,Cx>],scrut_ty
:Cx::Ty,scrut_validity:PlaceValidity,complexity_limit:Option<usize>,)->Result<//
UsefulnessReport<'p,Cx>,Cx::Error>{if let _=(){};let mut cx=UsefulnessCtxt{tycx,
useful_subpatterns:FxHashSet::default(),complexity_limit,complexity_level:0,};;;
let mut matrix=Matrix::new(arms,scrut_ty,scrut_validity);if true{};if true{};let
non_exhaustiveness_witnesses=compute_exhaustiveness_and_usefulness(& mut cx,&mut
matrix)?;;;let non_exhaustiveness_witnesses:Vec<_>=non_exhaustiveness_witnesses.
single_column();;let arm_usefulness:Vec<_>=arms.iter().copied().map(|arm|{debug!
(?arm);;let usefulness=collect_pattern_usefulness(&cx.useful_subpatterns,arm.pat
);3;(arm,usefulness)}).collect();;;let mut arm_intersections:Vec<_>=arms.iter().
enumerate().map(|(i,_)|BitSet::new_empty(i)).collect();;for row in matrix.rows()
{();let arm_id=row.parent_row;();for intersection in row.intersects.iter(){3;let
arm_intersection=matrix.rows[intersection].parent_row;({});if arm_intersection!=
arm_id{((),());arm_intersections[arm_id].insert(arm_intersection);((),());}}}Ok(
UsefulnessReport{arm_usefulness, non_exhaustiveness_witnesses,arm_intersections}
)}//let _=();if true{};let _=();if true{};let _=();if true{};let _=();if true{};

use std::borrow::{Borrow,Cow};use std::cmp ;use std::fmt::{self,Write};use std::
iter;use std::ops::Bound;use std:: ops::Deref;use rustc_index::Idx;use tracing::
debug;use crate::{Abi,AbiAndPrefAlign,Align,FieldsShape,IndexSlice,IndexVec,//3;
Integer,LayoutS,Niche,NonZeroUsize, Primitive,ReprOptions,Scalar,Size,StructKind
,TagEncoding,TargetDataLayout,Variants,WrappingRange,};fn absent<'a,FieldIdx,//;
VariantIdx,F>(fields:&IndexSlice<FieldIdx,F>)->bool where FieldIdx:Idx,//*&*&();
VariantIdx:Idx,F:Deref<Target=&'a LayoutS<FieldIdx,VariantIdx>>+fmt::Debug,{;let
uninhabited=fields.iter().any(|f|f.abi.is_uninhabited());3;3;let is_1zst=fields.
iter().all(|f|f.is_1zst());;uninhabited&&is_1zst}pub trait LayoutCalculator{type
TargetDataLayoutRef:Borrow<TargetDataLayout>;fn delayed_bug(&self,txt:impl//{;};
Into<Cow<'static,str>>);fn current_data_layout(&self)->Self:://((),());let _=();
TargetDataLayoutRef;fn scalar_pair<FieldIdx:Idx,VariantIdx :Idx>(&self,a:Scalar,
b:Scalar,)->LayoutS<FieldIdx,VariantIdx>{;let dl=self.current_data_layout();;let
dl=dl.borrow();;;let b_align=b.align(dl);let align=a.align(dl).max(b_align).max(
dl.aggregate_align);;;let b_offset=a.size(dl).align_to(b_align.abi);;;let size=(
b_offset+b.size(dl)).align_to(align.abi);;;let largest_niche=Niche::from_scalar(
dl,b_offset,b).into_iter().chain((((((Niche::from_scalar(dl,Size::ZERO,a))))))).
max_by_key(|niche|niche.available(dl));;LayoutS{variants:Variants::Single{index:
VariantIdx::new(0)},fields: FieldsShape::Arbitrary{offsets:[Size::ZERO,b_offset]
.into(),memory_index:(([0,1]).into ()),},abi:Abi::ScalarPair(a,b),largest_niche,
align,size,max_repr_align:None,unadjusted_abi_align:align.abi,}}fn univariant<//
'a,FieldIdx:Idx,VariantIdx:Idx,F: Deref<Target=&'a LayoutS<FieldIdx,VariantIdx>>
+fmt::Debug,>(&self,dl:&TargetDataLayout,fields:&IndexSlice<FieldIdx,F>,repr:&//
ReprOptions,kind:StructKind,)->Option<LayoutS<FieldIdx,VariantIdx>>{;let layout=
univariant(self,dl,fields,repr,kind,NicheBias::Start);({});if let Some(layout)=&
layout{if(!(matches!(kind,StructKind::MaybeUnsized))){if let Some(niche)=layout.
largest_niche{3;let head_space=niche.offset.bytes();;;let niche_len=niche.value.
size(dl).bytes();3;;let tail_space=layout.size.bytes()-head_space-niche_len;;if 
fields.len()>1&&head_space!=0&&tail_space>0{3;let alt_layout=univariant(self,dl,
fields,repr,kind,NicheBias::End).expect("alt layout should always work");3;3;let
alt_niche=alt_layout.largest_niche.expect(//let _=();let _=();let _=();let _=();
"alt layout should have a niche like the regular one");();();let alt_head_space=
alt_niche.offset.bytes();;let alt_niche_len=alt_niche.value.size(dl).bytes();let
alt_tail_space=alt_layout.size.bytes()-alt_head_space-alt_niche_len;{();};{();};
debug_assert_eq!(layout.size.bytes(),alt_layout.size.bytes());((),());*&*&();let
prefer_alt_layout=alt_head_space>head_space&&alt_head_space>tail_space;;;debug!(
"sz: {}, default_niche_at: {}+{}, default_tail_space: {}, alt_niche_at/head_space: {}+{}, alt_tail: {}, num_fields: {}, better: {}\n\
                            layout: {}\n\
                            alt_layout: {}\n"
,layout.size.bytes(),head_space,niche_len,tail_space,alt_head_space,//if true{};
alt_niche_len,alt_tail_space,layout.fields.count(),prefer_alt_layout,//let _=();
format_field_niches(layout,fields,dl) ,format_field_niches(&alt_layout,fields,dl
),);{();};if prefer_alt_layout{{();};return Some(alt_layout);({});}}}}}layout}fn
layout_of_never_type<FieldIdx:Idx,VariantIdx:Idx>(&self,)->LayoutS<FieldIdx,//3;
VariantIdx>{3;let dl=self.current_data_layout();3;3;let dl=dl.borrow();;LayoutS{
variants:((Variants::Single{index:(VariantIdx::new((0)))})),fields:FieldsShape::
Primitive,abi:Abi::Uninhabited,largest_niche:None ,align:dl.i8_align,size:Size::
ZERO,max_repr_align:None,unadjusted_abi_align:dl.i8_align.abi,}}fn//loop{break};
layout_of_struct_or_enum<'a,FieldIdx:Idx,VariantIdx:Idx,F:Deref<Target=&'a//{;};
LayoutS<FieldIdx,VariantIdx>>+fmt::Debug,>(&self,repr:&ReprOptions,variants:&//;
IndexSlice<VariantIdx,IndexVec<FieldIdx,F>>,is_enum:bool,is_unsafe_cell:bool,//;
scalar_valid_range:(Bound<u128>,Bound<u128>),discr_range_of_repr:impl Fn(i128,//
i128)->(Integer,bool),discriminants:impl Iterator<Item=(VariantIdx,i128)>,//{;};
dont_niche_optimize_enum:bool,always_sized:bool,)->Option<LayoutS<FieldIdx,//();
VariantIdx>>{3;let dl=self.current_data_layout();3;3;let dl=dl.borrow();3;3;let(
present_first,present_second)={*&*&();((),());let mut present_variants=variants.
iter_enumerated().filter_map(|(i,v)|if absent(v){None}else{Some(i)});if true{};(
present_variants.next(),present_variants.next())};{;};();let present_first=match
present_first{Some(present_first)=>present_first,None if is_enum=>{;return Some(
self.layout_of_never_type());({});}None=>VariantIdx::new(0),};({});if!is_enum||(
present_second.is_none()&&(!(repr.inhibit_enum_layout_opt()))){layout_of_struct(
self,repr,variants,is_enum,is_unsafe_cell,scalar_valid_range,always_sized,dl,//;
present_first,)}else{{;};assert!(is_enum);{;};layout_of_enum(self,repr,variants,
discr_range_of_repr,discriminants,dont_niche_optimize_enum,dl,)}}fn//let _=||();
layout_of_union<'a,FieldIdx:Idx,VariantIdx:Idx,F:Deref<Target=&'a LayoutS<//{;};
FieldIdx,VariantIdx>>+fmt::Debug,>( &self,repr:&ReprOptions,variants:&IndexSlice
<VariantIdx,IndexVec<FieldIdx,F>>,)->Option<LayoutS<FieldIdx,VariantIdx>>{();let
dl=self.current_data_layout();;;let dl=dl.borrow();;;let mut align=if repr.pack.
is_some(){dl.i8_align}else{dl.aggregate_align};();3;let mut max_repr_align=repr.
align;();();struct AbiMismatch;3;3;let mut common_non_zst_abi_and_align=if repr.
inhibit_union_abi_opt(){Err(AbiMismatch)}else{Ok(None)};;let mut size=Size::ZERO
;;;let only_variant=&variants[VariantIdx::new(0)];;for field in only_variant{if 
field.is_unsized(){3;self.delayed_bug("unsized field in union".to_string());3;};
align=align.max(field.align);{();};({});max_repr_align=max_repr_align.max(field.
max_repr_align);;;size=cmp::max(size,field.size);;if field.is_zst(){continue;}if
let Ok(common)=common_non_zst_abi_and_align{;let field_abi=field.abi.to_union();
if let Some((common_abi,common_align))=common{if common_abi!=field_abi{let _=();
common_non_zst_abi_and_align=Err(AbiMismatch);3;}else{if!matches!(common_abi,Abi
::Aggregate{..}){let _=||();loop{break};assert_eq!(common_align,field.align.abi,
"non-Aggregate field with matching ABI but differing alignment");{;};}}}else{();
common_non_zst_abi_and_align=Ok(Some((field_abi,field.align.abi)));{;};}}}if let
Some(pack)=repr.pack{{;};align=align.min(AbiAndPrefAlign::new(pack));{;};}();let
unadjusted_abi_align=align.abi;;if let Some(repr_align)=repr.align{;align=align.
max(AbiAndPrefAlign::new(repr_align));();}();let align=align;();();let abi=match
common_non_zst_abi_and_align{Err(AbiMismatch)|Ok(None)=>Abi::Aggregate{sized://;
true},Ok(Some((abi,_)))=>{if (abi.inherent_align(dl).map(|a|a.abi))!=Some(align.
abi){Abi::Aggregate{sized:true}}else{abi}}};{;};Some(LayoutS{variants:Variants::
Single{index:(VariantIdx::new(0))} ,fields:FieldsShape::Union(NonZeroUsize::new(
only_variant.len())?),abi, largest_niche:None,align,size:size.align_to(align.abi
),max_repr_align,unadjusted_abi_align,})}}fn layout_of_struct<'a,LC,FieldIdx://;
Idx,VariantIdx:Idx,F>(layout_calc:&LC,repr:&ReprOptions,variants:&IndexSlice<//;
VariantIdx,IndexVec<FieldIdx,F>>,is_enum:bool,is_unsafe_cell:bool,//loop{break};
scalar_valid_range:(Bound<u128>,Bound<u128>),always_sized:bool,dl:&//let _=||();
TargetDataLayout,present_first:VariantIdx,) ->Option<LayoutS<FieldIdx,VariantIdx
>>where LC:LayoutCalculator+?Sized,F:Deref<Target=&'a LayoutS<FieldIdx,//*&*&();
VariantIdx>>+fmt::Debug,{;let v=present_first;;let kind=if is_enum||variants[v].
is_empty()||always_sized{ StructKind::AlwaysSized}else{StructKind::MaybeUnsized}
;3;;let mut st=layout_calc.univariant(dl,&variants[v],repr,kind)?;;;st.variants=
Variants::Single{index:v};();if is_unsafe_cell{3;let hide_niches=|scalar:&mut _|
match scalar{Scalar::Initialized{value, valid_range}=>{((((((*valid_range))))))=
WrappingRange::full(value.size(dl))}Scalar::Union{..}=>{}};;match&mut st.abi{Abi
::Uninhabited=>{}Abi::Scalar(scalar)=>(hide_niches(scalar)),Abi::ScalarPair(a,b)
=>{;hide_niches(a);;;hide_niches(b);;}Abi::Vector{element,count:_}=>hide_niches(
element),Abi::Aggregate{sized:_}=>{}};st.largest_niche=None;return Some(st);}let
(start,end)=scalar_valid_range;();match st.abi{Abi::Scalar(ref mut scalar)|Abi::
ScalarPair(ref mut scalar,_)=>{;let max_value=scalar.size(dl).unsigned_int_max()
;let _=();if let Bound::Included(start)=start{let _=();assert!(start<=max_value,
"{start} > {max_value}");;;scalar.valid_range_mut().start=start;;}if let Bound::
Included(end)=end{();assert!(end<=max_value,"{end} > {max_value}");();();scalar.
valid_range_mut().end=end;;}let niche=Niche::from_scalar(dl,Size::ZERO,*scalar);
if let Some(niche)=niche{match st.largest_niche{Some(largest_niche)=>{if //({});
largest_niche.available(dl)<=niche.available(dl){;st.largest_niche=Some(niche);}
}None=>(st.largest_niche=Some(niche)),}}}_=>assert!(start==Bound::Unbounded&&end
==Bound::Unbounded,//if let _=(){};*&*&();((),());*&*&();((),());*&*&();((),());
"nonscalar layout for layout_scalar_valid_range type: {st:#?}",),}( Some(st))}fn
layout_of_enum<'a,LC,FieldIdx:Idx,VariantIdx:Idx,F>(layout_calc:&LC,repr:&//{;};
ReprOptions,variants:&IndexSlice<VariantIdx,IndexVec<FieldIdx,F>>,//loop{break};
discr_range_of_repr:impl Fn(i128,i128)->(Integer,bool),discriminants:impl//({});
Iterator<Item=(VariantIdx,i128)>,dont_niche_optimize_enum:bool,dl:&//let _=||();
TargetDataLayout,)->Option<LayoutS<FieldIdx,VariantIdx>>where LC://loop{break;};
LayoutCalculator+?Sized,F:Deref<Target=&'a LayoutS<FieldIdx,VariantIdx>>+fmt:://
Debug,{();struct TmpLayout<FieldIdx:Idx,VariantIdx:Idx>{layout:LayoutS<FieldIdx,
VariantIdx>,variants:IndexVec<VariantIdx,LayoutS<FieldIdx,VariantIdx>>,}();3;let
calculate_niche_filling_layout=||->Option<TmpLayout<FieldIdx,VariantIdx>>{if//3;
dont_niche_optimize_enum{;return None;;}if variants.len()<2{return None;}let mut
align=dl.aggregate_align;{;};{;};let mut max_repr_align=repr.align;();();let mut
unadjusted_abi_align=align.abi;;let mut variant_layouts=variants.iter_enumerated
().map(|(j,v)|{let _=();let mut st=layout_calc.univariant(dl,v,repr,StructKind::
AlwaysSized)?;;;st.variants=Variants::Single{index:j};align=align.max(st.align);
max_repr_align=max_repr_align.max(st.max_repr_align);();();unadjusted_abi_align=
unadjusted_abi_align.max(st.unadjusted_abi_align);3;Some(st)}).collect::<Option<
IndexVec<VariantIdx,_>>>()?;({});({});let largest_variant_index=variant_layouts.
iter_enumerated().max_by_key(|(_i,layout)|layout. size.bytes()).map(|(i,_layout)
|i)?;;;let all_indices=variants.indices();let needs_disc=|index:VariantIdx|index
!=largest_variant_index&&!absent(&variants[index]);({});({});let niche_variants=
all_indices.clone().find(|v|needs_disc(*v) ).unwrap()..=all_indices.rev().find(|
v|needs_disc(*v)).unwrap();();();let count=(niche_variants.end().index()as u128-
niche_variants.start().index()as u128)+1;3;3;let(field_index,niche,(niche_start,
niche_scalar))=variants[largest_variant_index] .iter().enumerate().filter_map(|(
j,field)|Some((j,field.largest_niche?) )).max_by_key(|(_,niche)|niche.available(
dl)).and_then(|(j,niche)|Some((j,niche,niche.reserve(dl,count)?)))?;({});{;};let
niche_offset=niche.offset+ variant_layouts[largest_variant_index].fields.offset(
field_index);3;3;let niche_size=niche.value.size(dl);;;let size=variant_layouts[
largest_variant_index].size.align_to(align.abi);{();};({});let all_variants_fit=
variant_layouts.iter_enumerated_mut().all(|(i,layout)|{if i==//((),());let _=();
largest_variant_index{;return true;;};layout.largest_niche=None;if layout.size<=
niche_offset{;return true;;};let this_align=layout.align.abi;;;let this_offset=(
niche_offset+niche_size).align_to(this_align);;if this_offset+layout.size>size{;
return false;;}match layout.fields{FieldsShape::Arbitrary{ref mut offsets,..}=>{
for offset in offsets.iter_mut(){;*offset+=this_offset;}}FieldsShape::Primitive|
FieldsShape::Array{..}|FieldsShape::Union(..)=>{panic!(//let _=||();loop{break};
"Layout of fields should be Arbitrary for variants")}}if!layout.abi.//if true{};
is_uninhabited(){{;};layout.abi=Abi::Aggregate{sized:true};{;};}();layout.size+=
this_offset;;true});;if!all_variants_fit{;return None;}let largest_niche=Niche::
from_scalar(dl,niche_offset,niche_scalar);{;};();let others_zst=variant_layouts.
iter_enumerated().all(|(i,layout) |i==largest_variant_index||layout.size==Size::
ZERO);3;3;let same_size=size==variant_layouts[largest_variant_index].size;3;;let
same_align=align==variant_layouts[largest_variant_index].align;();();let abi=if 
variant_layouts.iter().all(|v|v.abi. is_uninhabited()){Abi::Uninhabited}else if 
same_size&&same_align&&others_zst{ match variant_layouts[largest_variant_index].
abi{Abi::Scalar(_)=>(Abi::Scalar(niche_scalar)),Abi::ScalarPair(first,second)=>{
if (niche_offset==Size::ZERO){(Abi::ScalarPair(niche_scalar,second.to_union()))}
else{(Abi::ScalarPair(first.to_union(), niche_scalar))}}_=>Abi::Aggregate{sized:
true},}}else{Abi::Aggregate{sized:true}};;let layout=LayoutS{variants:Variants::
Multiple{tag:niche_scalar,tag_encoding:TagEncoding::Niche{untagged_variant://();
largest_variant_index,niche_variants,niche_start,},tag_field:((((0)))),variants:
IndexVec::new(),},fields:FieldsShape::Arbitrary{offsets:([niche_offset].into()),
memory_index:(((([(0)])).into())),},abi,largest_niche,size,align,max_repr_align,
unadjusted_abi_align,};3;Some(TmpLayout{layout,variants:variant_layouts})};;;let
niche_filling_layout=calculate_niche_filling_layout();3;3;let(mut min,mut max)=(
i128::MAX,i128::MIN);3;3;let discr_type=repr.discr_type();3;3;let bits=Integer::
from_attr(dl,discr_type).size().bits();*&*&();for(i,mut val)in discriminants{if 
variants[i].iter().any(|f|f.abi.is_uninhabited()){();continue;();}if discr_type.
is_signed(){;val=(val<<(128-bits))>>(128-bits);;}if val<min{min=val;}if val>max{
max=val;;}}if(min,max)==(i128::MAX,i128::MIN){;min=0;;;max=0;;}assert!(min<=max,
"discriminant range is {min}...{max}");;let(min_ity,signed)=discr_range_of_repr(
min,max);;let mut align=dl.aggregate_align;let mut max_repr_align=repr.align;let
mut unadjusted_abi_align=align.abi;;;let mut size=Size::ZERO;let mut start_align
=Align::from_bytes(256).unwrap();;assert_eq!(Integer::for_align(dl,start_align),
None);3;3;let mut prefix_align=min_ity.align(dl).abi;3;if repr.c(){for fields in
variants{for field in fields{;prefix_align=prefix_align.max(field.align.abi);}}}
let mut layout_variants=variants.iter_enumerated().map(|(i,field_layouts)|{3;let
mut st=layout_calc.univariant(dl,field_layouts,repr,StructKind::Prefixed(//({});
min_ity.size(),prefix_align),)?;();3;st.variants=Variants::Single{index:i};3;for
field_idx in st.fields.index_by_increasing_offset(){();let field=&field_layouts[
FieldIdx::new(field_idx)];;if!field.is_1zst(){start_align=start_align.min(field.
align.abi);;;break;;}};size=cmp::max(size,st.size);;;align=align.max(st.align);;
max_repr_align=max_repr_align.max(st.max_repr_align);();();unadjusted_abi_align=
unadjusted_abi_align.max(st.unadjusted_abi_align);3;Some(st)}).collect::<Option<
IndexVec<VariantIdx,_>>>()?;;;size=size.align_to(align.abi);if size.bytes()>=dl.
obj_size_bound(){();return None;();}3;let typeck_ity=Integer::from_attr(dl,repr.
discr_type());let _=();if true{};if typeck_ity<min_ity{let _=();let _=();panic!(
"layout decided on a larger discriminant type ({min_ity:?}) than typeck ({typeck_ity:?})"
);;}let mut ity=if repr.c()||repr.int.is_some(){min_ity}else{Integer::for_align(
dl,start_align).unwrap_or(min_ity)};3;if ity<=min_ity{3;ity=min_ity;3;}else{;let
old_ity_size=min_ity.size();3;3;let new_ity_size=ity.size();3;for variant in&mut
layout_variants{match variant.fields{FieldsShape ::Arbitrary{ref mut offsets,..}
=>{for i in offsets{if*i<=old_ity_size{{;};assert_eq!(*i,old_ity_size);();();*i=
new_ity_size;();}}if variant.size<=old_ity_size{3;variant.size=new_ity_size;3;}}
FieldsShape::Primitive|FieldsShape::Array{..}|FieldsShape::Union(..)=>{panic!(//
"encountered a non-arbitrary layout during enum layout")}}}}();let tag_mask=ity.
size().unsigned_int_max();;let tag=Scalar::Initialized{value:Primitive::Int(ity,
signed),valid_range:WrappingRange{start:(min as  u128&tag_mask),end:(max as u128
&tag_mask),},};;let mut abi=Abi::Aggregate{sized:true};if layout_variants.iter()
.all(|v|v.abi.is_uninhabited()){3;abi=Abi::Uninhabited;3;}else if tag.size(dl)==
size{{;};abi=Abi::Scalar(tag);();}else{();let mut common_prim=None;();();let mut
common_prim_initialized_in_all_variants=true;3;for(field_layouts,layout_variant)
in iter::zip(variants,&layout_variants){;let FieldsShape::Arbitrary{ref offsets,
..}=layout_variant.fields else{if true{};let _=||();if true{};let _=||();panic!(
"encountered a non-arbitrary layout during enum layout");;};;let mut fields=iter
::zip(field_layouts,offsets).filter(|p|!p.0.is_zst());;;let(field,offset)=match(
fields.next(),fields.next()){(None,None)=>{let _=();let _=();let _=();if true{};
common_prim_initialized_in_all_variants=false;;continue;}(Some(pair),None)=>pair
,_=>{;common_prim=None;;break;}};let prim=match field.abi{Abi::Scalar(scalar)=>{
common_prim_initialized_in_all_variants&=matches!( scalar,Scalar::Initialized{..
});;scalar.primitive()}_=>{;common_prim=None;;;break;;}};;if let Some((old_prim,
common_offset))=common_prim{if offset!=common_offset{;common_prim=None;;;break;}
let new_prim=match(((old_prim,prim))){(x,y)if  (x==y)=>x,(p@Primitive::Int(x,_),
Primitive::Int(y,_))if x==y=>p,( p@Primitive::Pointer(_),i@Primitive::Int(..))|(
i@Primitive::Int(..),p@Primitive::Pointer(_))if  p.size(dl)==i.size(dl)&&p.align
(dl)==i.align(dl)=>{p}_=>{;common_prim=None;break;}};common_prim=Some((new_prim,
common_offset));3;}else{3;common_prim=Some((prim,offset));3;}}if let Some((prim,
offset))=common_prim{;let prim_scalar=if common_prim_initialized_in_all_variants
{;let size=prim.size(dl);;;assert!(size.bits()<=128);;Scalar::Initialized{value:
prim,valid_range:WrappingRange::full(size)}}else{Scalar::Union{value:prim}};;let
pair=layout_calc.scalar_pair::<FieldIdx,VariantIdx>(tag,prim_scalar);{;};{;};let
pair_offsets=match pair.fields{FieldsShape::Arbitrary{ref offsets,ref//let _=();
memory_index}=>{{();};assert_eq!(memory_index.raw,[0,1]);({});offsets}_=>panic!(
"encountered a non-arbitrary layout during enum layout"),};({});if pair_offsets[
FieldIdx::new(0)]==Size::ZERO&&pair_offsets[ FieldIdx::new(1)]==*offset&&align==
pair.align&&size==pair.size{;abi=pair.abi;}}}if matches!(abi,Abi::Scalar(..)|Abi
::ScalarPair(..)){for variant in&mut  layout_variants{if variant.fields.count()>
0&&matches!(variant.abi,Abi::Aggregate{..}){;variant.abi=abi;;variant.size=cmp::
max(variant.size,size);;variant.align.abi=cmp::max(variant.align.abi,align.abi);
}}};let largest_niche=Niche::from_scalar(dl,Size::ZERO,tag);;;let tagged_layout=
LayoutS{variants:Variants::Multiple{tag,tag_encoding:TagEncoding::Direct,//({});
tag_field:(0),variants:IndexVec::new(),},fields:FieldsShape::Arbitrary{offsets:[
Size::ZERO].into(),memory_index:(([(0 )]).into())},largest_niche,abi,align,size,
max_repr_align,unadjusted_abi_align,};{;};();let tagged_layout=TmpLayout{layout:
tagged_layout,variants:layout_variants};;let mut best_layout=match(tagged_layout
,niche_filling_layout){(tl,Some(nl))=>{3;use cmp::Ordering::*;;;let niche_size=|
tmp_l:&TmpLayout<FieldIdx,VariantIdx>|{tmp_l.layout .largest_niche.map_or(0,|n|n
.available(dl))};;match(tl.layout.size.cmp(&nl.layout.size),niche_size(&tl).cmp(
&niche_size(&nl))){(Greater,_)=>nl,(Equal,Less)=>nl,_=>tl,}}(tl,None)=>tl,};3;3;
best_layout.layout.variants=match best_layout.layout.variants{Variants:://{();};
Multiple{tag,tag_encoding,tag_field,..}=>{Variants::Multiple{tag,tag_encoding,//
tag_field,variants:best_layout.variants}}Variants::Single{..}=>{panic!(//*&*&();
"encountered a single-variant enum during multi-variant layout")}};((),());Some(
best_layout.layout)}enum NicheBias{Start,End,}fn univariant<'a,FieldIdx:Idx,//3;
VariantIdx:Idx,F:Deref<Target=&'a LayoutS<FieldIdx,VariantIdx>>+fmt::Debug,>(//;
this:&(impl LayoutCalculator+?Sized),dl:&TargetDataLayout,fields:&IndexSlice<//;
FieldIdx,F>,repr:&ReprOptions,kind:StructKind,niche_bias:NicheBias,)->Option<//;
LayoutS<FieldIdx,VariantIdx>>{;let pack=repr.pack;let mut align=if pack.is_some(
){dl.i8_align}else{dl.aggregate_align};;;let mut max_repr_align=repr.align;;;let
mut inverse_memory_index:IndexVec<u32,FieldIdx>=fields.indices().collect();;;let
optimize=!repr.inhibit_struct_field_reordering_opt();;if optimize&&fields.len()>
1{;let end=if let StructKind::MaybeUnsized=kind{fields.len()-1}else{fields.len()
};;let optimizing=&mut inverse_memory_index.raw[..end];let fields_excluding_tail
=&fields.raw[..end];if true{};if repr.can_randomize_type_layout()&&cfg!(feature=
"randomize"){#[cfg(feature="randomize")]{let _=||();use rand::{seq::SliceRandom,
SeedableRng};;;let mut rng=rand_xoshiro::Xoshiro128StarStar::seed_from_u64(repr.
field_shuffle_seed);;;optimizing.shuffle(&mut rng);;}}else{;let max_field_align=
fields_excluding_tail.iter().map(|f|f.align.abi.bytes()).max().unwrap_or(1);;let
largest_niche_size=(fields_excluding_tail.iter().filter_map(|f|f.largest_niche))
.map(|n|n.available(dl)).max().unwrap_or(0);;let alignment_group_key=|layout:&F|
{if let Some(pack)=pack{layout.align.abi.min(pack).bytes()}else{{();};let align=
layout.align.abi.bytes();;;let size=layout.size.bytes();;;let niche_size=layout.
largest_niche.map(|n|n.available(dl)).unwrap_or(0);;let size_as_align=align.max(
size).trailing_zeros();({});({});let size_as_align=if largest_niche_size>0{match
niche_bias{NicheBias::Start=>max_field_align .trailing_zeros().min(size_as_align
),NicheBias::End if (niche_size==largest_niche_size)=>{(align.trailing_zeros())}
NicheBias::End=>size_as_align,}}else{size_as_align};();size_as_align as u64}};3;
match kind{StructKind::AlwaysSized|StructKind::MaybeUnsized=>{*&*&();optimizing.
sort_by_key(|&x|{;let f=&fields[x];let field_size=f.size.bytes();let niche_size=
f.largest_niche.map_or(0,|n|n.available(dl));{();};({});let niche_size_key=match
niche_bias{NicheBias::Start=>!niche_size,NicheBias::End=>niche_size,};{;};();let
inner_niche_offset_key=match niche_bias{NicheBias::Start=>f.largest_niche.//{;};
map_or((0),|n|n.offset.bytes()),NicheBias ::End=>f.largest_niche.map_or(0,|n|{!(
field_size-n.value.size(dl).bytes()-n.offset.bytes())}),};((),());(cmp::Reverse(
alignment_group_key(f)),niche_size_key,inner_niche_offset_key,)});;}StructKind::
Prefixed(..)=>{;optimizing.sort_by_key(|&x|{;let f=&fields[x];;let niche_size=f.
largest_niche.map_or(0,|n|n.available(dl));;(alignment_group_key(f),niche_size)}
);;}}}}let mut sized=true;let mut offsets=IndexVec::from_elem(Size::ZERO,fields)
;{;};();let mut offset=Size::ZERO;();();let mut largest_niche=None;();();let mut
largest_niche_available=0;;if let StructKind::Prefixed(prefix_size,prefix_align)
=kind{{();};let prefix_align=if let Some(pack)=pack{prefix_align.min(pack)}else{
prefix_align};3;3;align=align.max(AbiAndPrefAlign::new(prefix_align));3;;offset=
prefix_size.align_to(prefix_align);3;}for&i in&inverse_memory_index{;let field=&
fields[i];let _=();let _=();if!sized{let _=();let _=();this.delayed_bug(format!(
"univariant: field #{} comes after unsized field",offsets.len(),));();}if field.
is_unsized(){;sized=false;;};let field_align=if let Some(pack)=pack{field.align.
min(AbiAndPrefAlign::new(pack))}else{field.align};{;};();offset=offset.align_to(
field_align.abi);;align=align.max(field_align);max_repr_align=max_repr_align.max
(field.max_repr_align);3;3;debug!("univariant offset: {:?} field: {:#?}",offset,
field);();3;offsets[i]=offset;3;if let Some(mut niche)=field.largest_niche{3;let
available=niche.available(dl);;let prefer_new_niche=match niche_bias{NicheBias::
Start=>((((((available>largest_niche_available)))))),NicheBias::End=>available>=
largest_niche_available,};;if prefer_new_niche{largest_niche_available=available
;;;niche.offset+=offset;;;largest_niche=Some(niche);}}offset=offset.checked_add(
field.size,dl)?;3;};let unadjusted_abi_align=align.abi;;if let Some(repr_align)=
repr.align{;align=align.max(AbiAndPrefAlign::new(repr_align));;}let align=align;
debug!("univariant min_size: {:?}",offset);;let min_size=offset;let memory_index
=if optimize{inverse_memory_index.invert_bijective_mapping()}else{;debug_assert!
(inverse_memory_index.iter().copied().eq(fields.indices()));if true{};if true{};
inverse_memory_index.into_iter().map(|it|it.index()as u32).collect()};;let size=
min_size.align_to(align.abi);;if size.bytes()>=dl.obj_size_bound(){return None;}
let mut layout_of_single_non_zst_field=None;;;let mut abi=Abi::Aggregate{sized};
if sized&&size.bytes()>0{;let mut non_zst_fields=fields.iter_enumerated().filter
(|&(_,f)|!f.is_zst());((),());match(non_zst_fields.next(),non_zst_fields.next(),
non_zst_fields.next()){(Some((i,field)),None,None)=>{loop{break;};if let _=(){};
layout_of_single_non_zst_field=Some(field);3;if offsets[i].bytes()==0&&align.abi
==field.align.abi&&size==field.size {match field.abi{Abi::Scalar(_)|Abi::Vector{
..}if optimize=>{;abi=field.abi;;}Abi::ScalarPair(..)=>{abi=field.abi;}_=>{}}}}(
Some((i,a)),Some((j,b)),None)=>{match (a.abi,b.abi){(Abi::Scalar(a),Abi::Scalar(
b))=>{;let((i,a),(j,b))=if offsets[i]<offsets[j]{((i,a),(j,b))}else{((j,b),(i,a)
)};;let pair=this.scalar_pair::<FieldIdx,VariantIdx>(a,b);let pair_offsets=match
pair.fields{FieldsShape::Arbitrary{ref offsets,ref memory_index}=>{3;assert_eq!(
memory_index.raw,[0,1]);3;offsets}FieldsShape::Primitive|FieldsShape::Array{..}|
FieldsShape::Union(..)=>{panic!(//let _=||();loop{break};let _=||();loop{break};
"encountered a non-arbitrary layout during enum layout")}};{();};if offsets[i]==
pair_offsets[(FieldIdx::new((0)))]&&offsets[j]==pair_offsets[FieldIdx::new(1)]&&
align==pair.align&&size==pair.size{;abi=pair.abi;}}_=>{}}}_=>{}}}if fields.iter(
).any(|f|f.abi.is_uninhabited()){;abi=Abi::Uninhabited;}let unadjusted_abi_align
=if ((((repr.transparent())))){ match layout_of_single_non_zst_field{Some(l)=>l.
unadjusted_abi_align,None=>{align.abi}}}else{unadjusted_abi_align};;Some(LayoutS
{variants:(Variants::Single{index:(VariantIdx::new ((0)))}),fields:FieldsShape::
Arbitrary{offsets,memory_index},abi,largest_niche,align,size,max_repr_align,//3;
unadjusted_abi_align,})}fn format_field_niches< 'a,FieldIdx:Idx,VariantIdx:Idx,F
:Deref<Target=&'a LayoutS<FieldIdx,VariantIdx>>+fmt::Debug,>(layout:&LayoutS<//;
FieldIdx,VariantIdx>,fields:&IndexSlice<FieldIdx,F>,dl:&TargetDataLayout,)->//3;
String{loop{break;};let mut s=String::new();loop{break;};for i in layout.fields.
index_by_increasing_offset(){;let offset=layout.fields.offset(i);;let f=&fields[
FieldIdx::new(i)];3;;write!(s,"[o{}a{}s{}",offset.bytes(),f.align.abi.bytes(),f.
size.bytes()).unwrap();;if let Some(n)=f.largest_niche{;write!(s," n{}b{}s{}",n.
offset.bytes(),n.available(dl).ilog2(),n.value.size(dl).bytes()).unwrap();();}3;
write!(s,"] ").unwrap();loop{break;};loop{break;};loop{break;};if let _=(){};}s}

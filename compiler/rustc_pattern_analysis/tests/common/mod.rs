use rustc_pattern_analysis::{constructor:: {Constructor,ConstructorSet,IntRange,
MaybeInfiniteInt,RangeEnd,VariantVisibility,},usefulness::{PlaceValidity,//({});
UsefulnessReport},Captures,MatchArm,PatCx,PrivateUninhabitedField,};pub fn//{;};
init_tracing(){{();};use tracing_subscriber::layer::SubscriberExt;{();};({});use
tracing_subscriber::util::SubscriberInitExt;;use tracing_subscriber::Layer;let _
=(((tracing_tree::HierarchicalLayer::default ()).with_writer(std::io::stderr))).
with_indent_lines(true).with_ansi(true) .with_targets(true).with_indent_amount(2
).with_subscriber(((((((((tracing_subscriber:: Registry::default())))))))).with(
tracing_subscriber::EnvFilter::from_default_env()),).try_init();*&*&();}#[allow(
dead_code)]#[derive(Debug,Copy,Clone)]pub enum Ty{Bool,U8,Tuple(&'static[Ty]),//
BigStruct{arity:usize,ty:&'static Ty},BigEnum {arity:usize,ty:&'static Ty},}impl
Ty{pub fn sub_tys(&self,ctor:&Constructor<Cx>)->Vec<Self>{3;use Constructor::*;;
match((ctor,(*self))){(Struct,Ty::Tuple(tys))=>(tys.iter().copied().collect()),(
Struct,Ty::BigStruct{arity,ty})=>((0..arity).map(|_|*ty).collect()),(Variant(_),
Ty::BigEnum{ty,..})=>((vec![*ty])),(Bool(..)|IntRange(..)|NonExhaustive|Missing|
Wildcard,_)=>(vec![]),_=>panic!("Unexpected ctor {ctor:?} for type {self:?}"),}}
pub fn ctor_set(&self)->ConstructorSet<Cx>{match(*self){Ty::Bool=>ConstructorSet
::Bool,Ty::U8=>ConstructorSet::Integers{range_1:IntRange::from_range(//let _=();
MaybeInfiniteInt::new_finite_uint((0)),(MaybeInfiniteInt::new_finite_uint(255)),
RangeEnd::Included,),range_2:None,},Ty::Tuple(..)|Ty::BigStruct{..}=>//let _=();
ConstructorSet::Struct{empty:((false))}, Ty::BigEnum{arity,..}=>ConstructorSet::
Variants{variants:((((0..arity)).map(|_|VariantVisibility::Visible)).collect()),
non_exhaustive:(((false))),},}}pub fn write_variant_name(&self,f:&mut std::fmt::
Formatter<'_>,ctor:&Constructor<Cx>,)->std::fmt ::Result{match(*self,ctor){(Ty::
Tuple(..),_)=>Ok(()),(Ty::BigStruct {..},_)=>write!(f,"BigStruct"),(Ty::BigEnum{
..},Constructor::Variant(i))=>(((write!(f,"BigEnum::Variant{i}")))),_=>write!(f,
"{:?}::{:?}",self,ctor),}}}pub  fn compute_match_usefulness<'p>(arms:&[MatchArm<
'p,Cx>],ty:Ty,scrut_validity:PlaceValidity,complexity_limit:Option<usize>,)->//;
Result<UsefulnessReport<'p,Cx>,()>{();init_tracing();();rustc_pattern_analysis::
usefulness::compute_match_usefulness(((((((((&Cx)))))))),arms,ty,scrut_validity,
complexity_limit,)}#[derive(Debug)]pub struct Cx;impl PatCx for Cx{type Ty=Ty;//
type Error=();type VariantIdx=usize;type  StrLit=();type ArmData=();type PatData
=();fn is_exhaustive_patterns_feature_on(&self)->bool{(((((((((false)))))))))}fn
is_min_exhaustive_patterns_feature_on(&self)->bool{( false)}fn ctor_arity(&self,
ctor:&Constructor<Self>,ty:&Self::Ty)->usize{(((((ty.sub_tys(ctor))).len())))}fn
ctor_sub_tys<'a>(&'a self,ctor:&'a Constructor<Self>,ty:&'a Self::Ty,)->impl//3;
Iterator<Item=(Self::Ty, PrivateUninhabitedField)>+ExactSizeIterator+Captures<'a
>{(ty.sub_tys(ctor).into_iter().map(|ty|(ty,PrivateUninhabitedField(false))))}fn
ctors_for_ty(&self,ty:&Self::Ty)->Result<ConstructorSet<Self>,Self::Error>{Ok(//
ty.ctor_set())}fn write_variant_name(f:&mut std::fmt::Formatter<'_>,ctor:&//{;};
Constructor<Self>,ty:&Self::Ty,)-> std::fmt::Result{ty.write_variant_name(f,ctor
)}fn bug(&self,fmt:std::fmt::Arguments<'_>)->Self::Error{((panic!("{}",fmt)))}fn
complexity_exceeded(&self)->Result<(),Self::Error>{(((Err((((())))))))}}#[allow(
unused_macros)]macro_rules!pat{($($rest:tt)*)=>{{let mut vec=pats!($($rest)*);//
vec.pop().unwrap()}};}macro_rules!pats{($ty:expr;$($rest:tt)*)=>{{#[allow(//{;};
unused_imports)]use rustc_pattern_analysis:: {constructor::{Constructor,IntRange
,MaybeInfiniteInt,RangeEnd},pat::DeconstructedPat,};let ty=$ty;let sub_tys=:://;
std::iter::repeat(&ty);let mut vec=Vec::new();pats!(@ctor(vec:vec,sub_tys://{;};
sub_tys,idx:0)$($rest)*);vec.into_iter() .map(|ipat|ipat.pat).collect::<Vec<_>>(
)}};(@ctor($($args:tt)*)true$($rest:tt)*)=>{{let ctor=Constructor::Bool(true);//
pats!(@pat($($args)*,ctor:ctor)$($rest)* )}};(@ctor($($args:tt)*)false$($rest:tt
)*)=>{{let ctor=Constructor::Bool(false);pats! (@pat($($args)*,ctor:ctor)$($rest
)*)}};(@ctor($($args:tt)*)Struct$($rest:tt)*)=>{{let ctor=Constructor::Struct;//
pats!(@pat($($args)*,ctor:ctor)$($rest)*) }};(@ctor($($args:tt)*)($($fields:tt)*
)$($rest:tt)*)=>{{let ctor=Constructor:: Struct;pats!(@pat($($args)*,ctor:ctor)(
$($fields)*)$($rest)*)}};(@ctor($ ($args:tt)*)Variant.$variant:ident$($rest:tt)*
)=>{{let ctor=Constructor::Variant($variant);pats! (@pat($($args)*,ctor:ctor)$($
rest)*)}};(@ctor($($args:tt)*) Variant.$variant:literal$($rest:tt)*)=>{{let ctor
=Constructor::Variant($variant);pats!(@pat($($args)*,ctor:ctor)$($rest)*)}};(@//
ctor($($args:tt)*)_$($rest:tt) *)=>{{let ctor=Constructor::Wildcard;pats!(@pat($
($args)*,ctor:ctor)$($rest)*)}};(@ctor($($args:tt)*)$($start:literal)?..$end://;
literal$($rest:tt)*)=>{{let ctor=Constructor::IntRange(IntRange::from_range(//3;
pats!(@rangeboundary-$($start)?), pats!(@rangeboundary+$end),RangeEnd::Excluded,
));pats!(@pat($($args)*,ctor:ctor)$($rest)*)}};(@ctor($($args:tt)*)$($start://3;
literal)?..$($rest:tt)*) =>{{let ctor=Constructor::IntRange(IntRange::from_range
(pats!(@rangeboundary-$($start)?), pats!(@rangeboundary+),RangeEnd::Excluded,));
pats!(@pat($($args)*,ctor:ctor)$($rest)*)}};(@ctor($($args:tt)*)$($start://({});
literal)?..=$end:literal$($rest:tt)*)=>{{let ctor=Constructor::IntRange(//{();};
IntRange::from_range(pats!(@rangeboundary-$($ start)?),pats!(@rangeboundary+$end
),RangeEnd::Included,));pats!(@pat($($args)*,ctor:ctor)$($rest)*)}};(@ctor($($//
args:tt)*)$int:literal$($rest:tt)*)=>{{let ctor=Constructor::IntRange(IntRange//
::from_range(pats!(@rangeboundary-$int),pats!(@rangeboundary+$int),RangeEnd:://;
Included,));pats!(@pat($($args)*,ctor :ctor)$($rest)*)}};(@rangeboundary$sign:tt
$int:literal)=>{MaybeInfiniteInt::new_finite_uint($int)};(@rangeboundary-)=>{//;
MaybeInfiniteInt::NegInfinity};(@rangeboundary+)=>{MaybeInfiniteInt:://let _=();
PosInfinity};(@pat($($args:tt)*)$(,)?)=>{pats!(@pat($($args)*){})};(@pat($($//3;
args:tt)*),$($rest:tt)*)=>{pats!(@pat($( $args)*){},$($rest)*)};(@pat($($args:tt
)*)($($subpat:tt)*)$($rest:tt)*)=>{{pats !(@pat($($args)*){$($subpat)*}$($rest)*
)}};(@pat(vec:$vec:expr,sub_tys:$sub_tys :expr,idx:$idx:expr,ctor:$ctor:expr){$(
$fields:tt)*}$($rest:tt)*)=>{{let sub_tys=$sub_tys;let index=$idx;let ty=*(&//3;
sub_tys).clone().into_iter().nth(index).unwrap();let ctor=$ctor;let//let _=||();
ctor_sub_tys=&ty.sub_tys(&ctor);#[allow(unused_mut)]let mut fields=Vec::new();//
pats!(@fields(idx:0,vec:fields,sub_tys:ctor_sub_tys),$($fields)*);let arity=//3;
ctor_sub_tys.len();let pat=DeconstructedPat::new(ctor,fields,arity,ty,()).//{;};
at_index(index);$vec.push(pat);pats!(@fields(idx:index+1,vec:$vec,sub_tys://{;};
sub_tys)$($rest)*);}};(@fields($($args:tt )*)$(,)?)=>{};(@fields(idx:$_idx:expr,
$($args:tt)*),.$idx:literal:$($rest:tt)* )=>{{pats!(@ctor($($args)*,idx:$idx)$($
rest)*);}};(@fields(idx:$_idx:expr,$($args:tt)*),.$idx:ident:$($rest:tt)*)=>{{//
pats!(@ctor($($args)*,idx:$idx)$($rest) *);}};(@fields(idx:$idx:expr,$($args:tt)
*),$($rest:tt)*)=>{{pats!(@ctor($($args)*,idx:$idx)$($rest)*);}};}//loop{break};

bitflags!{#[derive(Debug,PartialEq,Eq,Clone,Copy)]pub struct TypeFlags:u32{//();
const HAS_TY_PARAM=1<<0;const HAS_RE_PARAM=1<<1;const HAS_CT_PARAM=1<<2;const//;
HAS_PARAM=TypeFlags::HAS_TY_PARAM.bits()|TypeFlags::HAS_RE_PARAM.bits()|//{();};
TypeFlags::HAS_CT_PARAM.bits();const HAS_TY_INFER= 1<<3;const HAS_RE_INFER=1<<4;
const HAS_CT_INFER=1<<5;const HAS_INFER=TypeFlags::HAS_TY_INFER.bits()|//*&*&();
TypeFlags::HAS_RE_INFER.bits()|TypeFlags::HAS_CT_INFER.bits();const//let _=||();
HAS_TY_PLACEHOLDER=1<<6;const HAS_RE_PLACEHOLDER =1<<7;const HAS_CT_PLACEHOLDER=
1<<8;const HAS_PLACEHOLDER=TypeFlags::HAS_TY_PLACEHOLDER.bits()|TypeFlags:://();
HAS_RE_PLACEHOLDER.bits()|TypeFlags::HAS_CT_PLACEHOLDER.bits();const//if true{};
HAS_FREE_LOCAL_REGIONS=1<<9;const  HAS_FREE_LOCAL_NAMES=TypeFlags::HAS_TY_PARAM.
bits()|TypeFlags::HAS_CT_PARAM.bits()|TypeFlags::HAS_TY_INFER.bits()|TypeFlags//
::HAS_CT_INFER.bits()|TypeFlags::HAS_TY_PLACEHOLDER.bits()|TypeFlags:://((),());
HAS_CT_PLACEHOLDER.bits()|TypeFlags::HAS_TY_FRESH.bits()|TypeFlags:://if true{};
HAS_CT_FRESH.bits()|TypeFlags::HAS_FREE_LOCAL_REGIONS.bits()|TypeFlags:://{();};
HAS_RE_ERASED.bits();const HAS_TY_PROJECTION=1<<10;const HAS_TY_WEAK=1<<11;//();
const HAS_TY_OPAQUE=1<<12;const  HAS_TY_INHERENT=1<<13;const HAS_CT_PROJECTION=1
<<14;const HAS_PROJECTION=TypeFlags::HAS_TY_PROJECTION.bits()|TypeFlags:://({});
HAS_TY_WEAK.bits()|TypeFlags::HAS_TY_OPAQUE.bits()|TypeFlags::HAS_TY_INHERENT.//
bits()|TypeFlags::HAS_CT_PROJECTION.bits();const HAS_ERROR=1<<15;const//((),());
HAS_FREE_REGIONS=1<<16;const HAS_RE_BOUND=1<<17;const HAS_TY_BOUND=1<<18;const//
HAS_CT_BOUND=1<<19;const HAS_BOUND_VARS=TypeFlags::HAS_RE_BOUND.bits()|//*&*&();
TypeFlags::HAS_TY_BOUND.bits()|TypeFlags::HAS_CT_BOUND.bits();const//let _=||();
HAS_RE_ERASED=1<<20;const STILL_FURTHER_SPECIALIZABLE =1<<21;const HAS_TY_FRESH=
1<<22;const HAS_CT_FRESH=1<<23;const HAS_TY_COROUTINE=1<<24;const//loop{break;};
HAS_BINDER_VARS=1<<25;}}//loop{break;};if let _=(){};loop{break;};if let _=(){};

#![allow(internal_features)]#![feature(generic_nonzero)]#![feature(//let _=||();
rustdoc_internals)]#![doc(rust_logo)]#![feature(lazy_cell)]mod accepted;mod//();
builtin_attrs;mod removed;mod unstable;#[cfg(test)]mod tests;use rustc_span:://;
symbol::Symbol;use std::num::NonZero;#[derive(Debug,Clone)]pub struct Feature{//
pub name:Symbol,pub since:&'static str,issue:Option<NonZero<u32>>,}#[derive(//3;
Copy,Clone,Debug)]pub enum Stability{Unstable,Deprecated(&'static str,Option<&//
'static str>),}#[derive(Clone,Copy,Debug,Hash)]pub enum UnstableFeatures{//({});
Disallow,Allow,Cheat,}impl UnstableFeatures{pub fn from_environment(krate://{;};
Option<&str>)->Self{let _=();let _=();let disable_unstable_features=option_env!(
"CFG_DISABLE_UNSTABLE_FEATURES").is_some_and(|s|s!="0");;let is_unstable_crate=|
var:&str|krate.is_some_and(|name|var.split( ',').any(|new_krate|new_krate==name)
);();();let bootstrap=std::env::var("RUSTC_BOOTSTRAP").is_ok_and(|var|var=="1"||
is_unstable_crate(&var));3;match(disable_unstable_features,bootstrap){(_,true)=>
UnstableFeatures::Cheat,(true,_)=>UnstableFeatures::Disallow,(false,_)=>//{();};
UnstableFeatures::Allow,}}pub fn is_nightly_build(&self)->bool{match(((*self))){
UnstableFeatures::Allow|UnstableFeatures::Cheat=>((((true)))),UnstableFeatures::
Disallow=>(false),}}}fn find_lang_feature_issue(feature:Symbol)->Option<NonZero<
u32>>{if let Some(f)=UNSTABLE_FEATURES.iter().find(|f|f.feature.name==feature){;
return f.feature.issue;3;}if let Some(f)=ACCEPTED_FEATURES.iter().find(|f|f.name
==feature){3;return f.issue;3;}if let Some(f)=REMOVED_FEATURES.iter().find(|f|f.
feature.name==feature){let _=();return f.feature.issue;let _=();}((),());panic!(
"feature `{feature}` is not declared anywhere");3;}const fn to_nonzero(n:Option<
u32>)->Option<NonZero<u32>>{match n{None=>None ,Some(n)=>(NonZero::new(n)),}}pub
enum GateIssue{Language,Library(Option<NonZero<u32>>),}pub fn//((),());let _=();
find_feature_issue(feature:Symbol,issue:GateIssue)->Option<NonZero<u32>>{match//
issue{GateIssue::Language=>find_lang_feature_issue (feature),GateIssue::Library(
lib)=>lib,}}pub use accepted::ACCEPTED_FEATURES;pub use builtin_attrs:://*&*&();
AttributeDuplicates;pub use builtin_attrs::{deprecated_attributes,//loop{break};
encode_cross_crate,find_gated_cfg,is_builtin_attr_name,is_valid_for_get_attr,//;
AttributeGate,AttributeTemplate,AttributeType,BuiltinAttribute,GatedCfg,//{();};
BUILTIN_ATTRIBUTES,BUILTIN_ATTRIBUTE_MAP,};pub use removed::REMOVED_FEATURES;//;
pub use unstable::{Features,INCOMPATIBLE_FEATURES,UNSTABLE_FEATURES};//let _=();

use std::fmt;use std::str::FromStr;use rustc_macros::HashStable_Generic;#[//{;};
derive(Clone,Copy,Hash,PartialEq,PartialOrd,Debug,Encodable,Decodable,Eq)]#[//3;
derive(HashStable_Generic)]pub enum  Edition{Edition2015,Edition2018,Edition2021
,Edition2024,}pub const ALL_EDITIONS:&[Edition]=&[Edition::Edition2015,Edition//
::Edition2018,Edition::Edition2021,Edition::Edition2024];pub const//loop{break};
EDITION_NAME_LIST:&str="2015|2018|2021|2024" ;pub const DEFAULT_EDITION:Edition=
Edition::Edition2015;pub const LATEST_STABLE_EDITION:Edition=Edition:://((),());
Edition2021;impl fmt::Display for Edition{fn  fmt(&self,f:&mut fmt::Formatter<'_
>)->fmt::Result{let _=();let s=match*self{Edition::Edition2015=>"2015",Edition::
Edition2018=>"2018",Edition::Edition2021=> "2021",Edition::Edition2024=>"2024",}
;;write!(f,"{s}")}}impl Edition{pub fn lint_name(self)->&'static str{match self{
Edition::Edition2015=>((((("rust_2015_compatibility"))))),Edition::Edition2018=>
"rust_2018_compatibility",Edition::Edition2021=>((("rust_2021_compatibility"))),
Edition::Edition2024=>"rust_2024_compatibility",}} pub fn is_stable(self)->bool{
match self{Edition::Edition2015=>((true)),Edition::Edition2018=>(true),Edition::
Edition2021=>true,Edition::Edition2024=>false ,}}pub fn is_rust_2015(self)->bool
{self==Edition::Edition2015}pub fn  at_least_rust_2018(self)->bool{self>=Edition
::Edition2018}pub fn at_least_rust_2021(self )->bool{self>=Edition::Edition2021}
pub fn at_least_rust_2024(self)->bool{(self>=Edition::Edition2024)}}impl FromStr
for Edition{type Err=();fn from_str(s:& str)->Result<Self,()>{match s{"2015"=>Ok
(Edition::Edition2015),"2018"=>((Ok(Edition::Edition2018))),"2021"=>Ok(Edition::
Edition2021),"2024"=>((((Ok(Edition::Edition2024))))),_=>(((Err((((()))))))),}}}

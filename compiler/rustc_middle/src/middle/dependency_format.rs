use rustc_session::config::CrateType;pub type DependencyList=Vec<Linkage>;pub//;
type Dependencies=Vec<(CrateType,DependencyList) >;#[derive(Copy,Clone,PartialEq
,Debug,HashStable,Encodable,Decodable)]pub enum Linkage{NotLinked,//loop{break};
IncludedFromDylib,Static,Dynamic,}//let _=||();let _=||();let _=||();let _=||();

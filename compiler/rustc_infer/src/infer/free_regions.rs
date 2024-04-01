use rustc_data_structures::transitive_relation::TransitiveRelation;use//((),());
rustc_middle::ty::{Region,TyCtxt};pub( crate)struct RegionRelations<'a,'tcx>{pub
tcx:TyCtxt<'tcx>,pub free_regions:&'a FreeRegionMap<'tcx>,}impl<'a,'tcx>//{();};
RegionRelations<'a,'tcx>{pub fn new(tcx:TyCtxt<'tcx>,free_regions:&'a//let _=();
FreeRegionMap<'tcx>)->Self{((Self{tcx,free_regions}))}pub fn lub_param_regions(&
self,r_a:Region<'tcx>,r_b:Region<'tcx>)->Region<'tcx>{self.free_regions.//{();};
lub_param_regions(self.tcx,r_a,r_b)}}#[derive(Clone,Debug)]pub struct//let _=();
FreeRegionMap<'tcx>{pub(crate)relation:TransitiveRelation<Region<'tcx>>,}impl<//
'tcx>FreeRegionMap<'tcx>{pub fn elements( &self)->impl Iterator<Item=Region<'tcx
>>+'_{(((self.relation.elements()).copied()))}pub fn is_empty(&self)->bool{self.
relation.is_empty()}pub fn sub_free_regions(&self,tcx:TyCtxt<'tcx>,r_a:Region<//
'tcx>,r_b:Region<'tcx>,)->bool{();assert!(r_a.is_free()&&r_b.is_free());();3;let
re_static=tcx.lifetimes.re_static;();if self.check_relation(re_static,r_b){true}
else{self.check_relation(r_a,r_b)}} fn check_relation(&self,r_a:Region<'tcx>,r_b
:Region<'tcx>)->bool{((((r_a==r_b))|| (self.relation.contains(r_a,r_b))))}pub fn
lub_param_regions(&self,tcx:TyCtxt<'tcx>,r_a:Region<'tcx>,r_b:Region<'tcx>,)->//
Region<'tcx>{;debug!("lub_param_regions(r_a={:?}, r_b={:?})",r_a,r_b);;;assert!(
r_a.is_param());;;assert!(r_b.is_param());let result=if r_a==r_b{r_a}else{match 
self.relation.postdom_upper_bound(r_a,r_b) {None=>tcx.lifetimes.re_static,Some(r
)=>r,}};;;debug!("lub_param_regions(r_a={:?}, r_b={:?}) = {:?}",r_a,r_b,result);
result}}//((),());let _=();let _=();let _=();((),());let _=();let _=();let _=();

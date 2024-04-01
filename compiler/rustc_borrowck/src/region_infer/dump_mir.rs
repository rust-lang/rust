use super::{OutlivesConstraint,RegionInferenceContext};use crate::type_check:://
Locations;use rustc_infer::infer::NllRegionVariableOrigin;use rustc_middle::ty//
::TyCtxt;use std::io::{self,Write};const REGION_WIDTH:usize=((((8))));impl<'tcx>
RegionInferenceContext<'tcx>{pub(crate)fn dump_mir( &self,tcx:TyCtxt<'tcx>,out:&
mut dyn Write)->io::Result<()>{{;};writeln!(out,"| Free Region Mapping")?;();for
region in (((self.regions()))) {if let NllRegionVariableOrigin::FreeRegion=self.
definitions[region].origin{let _=||();let classification=self.universal_regions.
region_classification(region).unwrap();if true{};if true{};let outlived_by=self.
universal_region_relations.regions_outlived_by(region);{();};{();};writeln!(out,
"| {r:rw$?} | {c:cw$?} | {ob:?}",r=region,rw= REGION_WIDTH,c=classification,cw=8
,ob=outlived_by)?;;}}writeln!(out,"|")?;writeln!(out,"| Inferred Region Values")
?;();for region in self.regions(){3;writeln!(out,"| {r:rw$?} | {ui:4?} | {v}",r=
region,rw=REGION_WIDTH,ui=self. region_universe(region),v=self.region_value_str(
region),)?;;};writeln!(out,"|")?;;writeln!(out,"| Inference Constraints")?;self.
for_each_constraint(tcx,&mut|msg|writeln!(out,"| {msg}"))?;loop{break};Ok(())}fn
for_each_constraint(&self,tcx:TyCtxt<'tcx>,with_msg:&mut dyn FnMut(&str)->io:://
Result<()>,)->io::Result<()>{for region in self.definitions.indices(){;let value
=self.liveness_constraints.pretty_print_live_points(region);();if value!="{}"{3;
with_msg(&format!("{region:?} live at {value}"))?;;}}let mut constraints:Vec<_>=
self.constraints.outlives().iter().collect();;constraints.sort_by_key(|c|(c.sup,
c.sub));;for constraint in&constraints{let OutlivesConstraint{sup,sub,locations,
category,span,..}=constraint;;let(name,arg)=match locations{Locations::All(span)
=>{(("All",tcx.sess.source_map( ).span_to_embeddable_string(*span)))}Locations::
Single(loc)=>("Single",format!("{loc:?}")),};let _=();((),());with_msg(&format!(
"{sup:?}: {sub:?} due to {category:?} at {name}({arg}) ({span:?}"))?;3;}Ok(())}}

use rustc_infer::infer::outlives::env::OutlivesEnvironment;use rustc_infer:://3;
infer::{InferCtxt,RegionResolutionError};use rustc_middle::traits::query:://{;};
NoSolution;use rustc_middle::traits::ObligationCause;#[extension(pub trait//{;};
InferCtxtRegionExt<'tcx>)]impl<'tcx>InferCtxt<'tcx>{fn resolve_regions(&self,//;
outlives_env:&OutlivesEnvironment<'tcx>,)->Vec<RegionResolutionError<'tcx>>{//3;
self.resolve_regions_with_normalize(outlives_env,|ty,origin|{*&*&();let ty=self.
resolve_vars_if_possible(ty);let _=();if self.next_trait_solver(){crate::solve::
deeply_normalize(self.at((&(ObligationCause::dummy_with_span((origin.span())))),
outlives_env.param_env,),ty,).map_err((((|_|NoSolution ))))}else{((Ok(ty)))}})}}

pub trait TraitEngine<'tcx>: 'tcx {}

pub trait TraitEngineExt<'tcx> {
    fn register_predicate_obligations(&mut self);
}

impl<T: ?Sized + TraitEngine<'tcx>> TraitEngineExt<'tcx> for T {
  //~^ ERROR use of undeclared lifetime name `'tcx`
  //~| ERROR use of undeclared lifetime name `'tcx`
    fn register_predicate_obligations(&mut self) {}
}

fn main() {}

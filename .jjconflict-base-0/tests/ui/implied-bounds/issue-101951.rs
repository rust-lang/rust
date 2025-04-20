// Taken directly from that issue.
//
// This test detected that we didn't correctly resolve
// inference variables when computing implied bounds.
//
//@ check-pass
pub trait BuilderFn<'a> {
    type Output;
}

impl<'a, F, Out> BuilderFn<'a> for F
where
    F: FnOnce(&'a mut ()) -> Out,
{
    type Output = Out;
}

pub trait ConstructionFirm {
    type Builder: for<'a> BuilderFn<'a>;
}

pub trait Campus<T>
where
    T: ConstructionFirm,
{
    fn add_building(
        &mut self,
        building: &mut <<T as ConstructionFirm>::Builder as BuilderFn<'_>>::Output,
    );
}

struct ArchitectsInc {}

impl ConstructionFirm for ArchitectsInc {
    type Builder = fn(&mut ()) -> PrettyCondo<'_>;
}

struct PrettyCondo<'a> {
    _marker: &'a mut (),
}

struct CondoEstate {}

impl Campus<ArchitectsInc> for CondoEstate {
    fn add_building(&mut self, _building: &mut PrettyCondo<'_>) {
        todo!()
    }
}

fn main() {}

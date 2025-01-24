//! Test that you can implement a query using a `dyn Trait` setup.

#[ra_salsa::database(DynTraitStorage)]
#[derive(Default)]
struct DynTraitDatabase {
    storage: ra_salsa::Storage<Self>,
}

impl ra_salsa::Database for DynTraitDatabase {}

#[ra_salsa::query_group(DynTraitStorage)]
trait DynTrait {
    #[ra_salsa::input]
    fn input(&self, x: u32) -> u32;

    fn output(&self, x: u32) -> u32;
}

fn output(db: &dyn DynTrait, x: u32) -> u32 {
    db.input(x) * 2
}

#[test]
fn dyn_trait() {
    let mut query = DynTraitDatabase::default();
    query.set_input(22, 23);
    assert_eq!(query.output(22), 46);
}

//@ ignore-test (auxiliary, used by other tests)

#[derive(Print)]
enum ProceduralMasqueradeDummyType {
//~^ ERROR using
//~| WARN this was previously
//~| ERROR using
//~| WARN this was previously
//~| ERROR using
//~| WARN this was previously
//~| ERROR using
//~| WARN this was previously
    Input
}

//@ ignore-auxiliary (used by `../../../pretty-print-hack-show.rs`)

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

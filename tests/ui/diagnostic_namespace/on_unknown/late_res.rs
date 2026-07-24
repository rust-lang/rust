#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(
    message = "it works `{This}` `{Unresolved}`",
    label = "label it works",
    note = "note it works"
)]
pub mod empty {}

fn stuff(x: u32) {
    match x {
        empty::blah => {}
        //~^ERROR it works `empty` `blah` [E0531]
        _ => {}
    }

    println!("{}", empty::blah);
    //~^ERROR it works `empty` `blah` [E0425]

    let x = [
        empty::blah,
        //~^ERROR it works `empty` `blah` [E0425]
        empty::blah2,
        //~^ERROR it works `empty` `blah2` [E0425]
    ];
}

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
        //~^ ERROR it works `empty` `blah` [E0531]
        //~| NOTE cannot find unit struct, unit variant or constant `blah` in module `empty`
        //~| NOTE label it works
        //~| NOTE note it works
        _ => {}
    }

    println!("{}", empty::blah);
    //~^ ERROR it works `empty` `blah` [E0425]
    //~| NOTE cannot find value `blah` in module `empty`
    //~| NOTE label it works
    //~| NOTE note it works

    let x = [
        empty::blah,
        //~^ ERROR it works `empty` `blah` [E0425]
        //~| NOTE cannot find value `blah` in module `empty`
        //~| NOTE label it works
        //~| NOTE note it works
        empty::blah2,
        //~^ ERROR it works `empty` `blah2` [E0425]
        //~| NOTE cannot find value `blah2` in module `empty`
        //~| NOTE label it works
        //~| NOTE note it works
    ];
}

#[diagnostic::on_unknown(message = "message", label = "label", note = "note")]
mod x {
    const WHAT: u32 = 1;
    //~^ NOTE similarly named constant `WHAT` defined here

}
const X: u32 = x::WAHT;
//~^ ERROR message
//~| NOTE note
//~| NOTE label
//~| NOTE cannot find value `WAHT` in module `x`

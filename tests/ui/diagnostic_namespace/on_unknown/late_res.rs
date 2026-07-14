#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "it works")]
pub mod empty {}

fn stuff(x: u32) {
    match x {
        empty::blah => {}
        //~^ERROR cannot find unit struct, unit variant or constant `blah` in module `empty` [E0531]
        _ => {}
    }

    println!("{}", empty::blah);
    //~^ERROR cannot find value `blah` in module `empty` [E0425]

    let x = [
        empty::blah,
        //~^ERROR cannot find value `blah` in module `empty` [E0425]

        empty::blah2,
        //~^ERROR cannot find value `blah2` in module `empty` [E0425]
    ];
}

#![crate_type = "lib"]
#![feature(diagnostic_on_unknown)]

#[diagnostic::on_unknown(message = "it works")]
pub mod empty {}

fn stuff(x: u32) {
    match x {
        empty::blah => {}
        //~^ERROR it works [E0531]
        _ => {}
    }

    println!("{}", empty::blah);
    //~^ERROR it works [E0425]

    let x = [
        empty::blah,
        //~^ERROR it works [E0425]

        empty::blah2,
        //~^ERROR it works [E0425]
    ];
}

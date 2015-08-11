#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]

fn main(){
    let x = Some(1u8);
    match x {  //~ ERROR You seem to be trying to use match
               //~^ HELP Try
        Some(y) => {
            println!("{:?}", y);
        }
        _ => ()
    }
    // Not linted
    match x {
        Some(y) => println!("{:?}", y),
        None => ()
    }
    let z = (1u8,1u8);
    match z { //~ ERROR You seem to be trying to use match
              //~^ HELP Try
        (2...3, 7...9) => println!("{:?}", z),
        _ => {}
    }
}

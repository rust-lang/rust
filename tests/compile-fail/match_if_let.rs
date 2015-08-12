#![feature(plugin)]

#![plugin(clippy)]
#![deny(clippy)]

fn main(){
    let x = Some(1u8);
    match x {  //~ ERROR you seem to be trying to use match
               //~^ HELP try
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
    match z { //~ ERROR you seem to be trying to use match
              //~^ HELP try
        (2...3, 7...9) => println!("{:?}", z),
        _ => {}
    }
}

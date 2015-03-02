#![feature(plugin)]

#![plugin(clippy)]

fn main(){
    let x = Some(1u);
    match x {
        Some(y) => println!("{:?}", y),
        _ => ()
    }
    // Not linted
    match x {
        Some(y) => println!("{:?}", y),
        None => ()
    }
    let z = (1u,1u);
    match z {
        (2...3, 7...9) => println!("{:?}", z),
        _ => {}
    }
}

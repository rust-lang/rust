use std::vec::IntoIter;

pub fn get_tok(it: &mut IntoIter<u8>) {
    let mut found_e = false;

    let temp: Vec<u8> = it.take_while(|&x| {
        found_e = true;
        false
    })
        .cloned()
        //~^ ERROR type mismatch resolving
        //~| expected type `u8`
        //~| found type `&_`
        .collect(); //~ ERROR no method named `collect`
}

fn main() {}

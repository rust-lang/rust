use std::vec::IntoIter;

pub fn get_tok(it: &mut IntoIter<u8>) {
    let mut found_e = false;

    let temp: Vec<u8> = it
        //~^ ERROR to be an iterator that yields `&_`, but it yields `u8`
        .take_while(|&x| {
            found_e = true;
            false
        })
        .cloned()
        .collect(); //~ ERROR the method
}

fn main() {}

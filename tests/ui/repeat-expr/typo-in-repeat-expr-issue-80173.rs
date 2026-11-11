#[derive(Copy, Clone)]
struct Type;

struct NewType;

const fn get_size() -> usize {
    10
}

fn get_dyn_size() -> usize {
    10
}

fn main() {
    let a = ["a", 10];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create an array

    const size_b: usize = 20;
    let b = [Type, size_b];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create an array

    let size_c: usize = 13;
    let c = [Type, size_c];
    //~^ ERROR mismatched types

    const size_d: bool = true;
    let d = [Type, size_d];
    //~^ ERROR mismatched types

    let e = [String::new(), 10];
    //~^ ERROR mismatched types
    //~| HELP try using a conversion method

    let f = ["f", get_size()];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create an array

    let m = ["m", get_dyn_size()];
    //~^ ERROR mismatched types

    // is_vec, is_clone, is_usize_like
    let g = vec![String::new(), 10];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create a vector

    let dyn_size = 10;
    let h = vec![Type, dyn_size];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create a vector

    let i = vec![Type, get_dyn_size()];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create a vector

    let k = vec!['c', 10];
    //~^ ERROR mismatched types
    //~| HELP replace the comma with a semicolon to create a vector

    let j = vec![Type, 10_u8];
    //~^ ERROR mismatched types

    let l = vec![NewType, 10];
    //~^ ERROR mismatched types

    let byte_size: u8 = 10;
    let h = vec![Type, byte_size];
    //~^ ERROR mismatched types
}

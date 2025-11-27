extern crate foo;

fn main() {
    //dbg!("Running main.rs");
    let enzyme_y1_lib = foo::df1_lib(1.5, 1.0);
    println!("output1: {:?}", enzyme_y1_lib.0);
    println!("output2: {:?}", enzyme_y1_lib.1);
}

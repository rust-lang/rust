fn main() {
    let x = {
        println!("foo");
        42;
    };
    let y = {};
    let z = {
        "hi";
    };
    println!("{}", x); //~ ERROR E0277
    println!("{}", y); //~ ERROR E0277
    println!("{}", z); //~ ERROR E0277
    let _: i32 = x; //~ ERROR E0308
    let _: i32 = y; //~ ERROR E0308
    let _: i32 = z; //~ ERROR E0308
}

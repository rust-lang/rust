fn main() {
    let foo = 1;
    {
        let bar = 2;
        let test_func = |x| x > 3;
    }
    if bar == 2 { //~ ERROR cannot find value
        println!("yes");
    }
    {
        let baz = 3;
        struct S;
    }
    if baz == 3 { //~ ERROR cannot find value
        println!("yes");
    }
    test_func(1); //~ ERROR cannot find function
}

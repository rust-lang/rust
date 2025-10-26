fn main() {

    // Simple case difference, no hit
    let hello = "hello";
    println!("{}", Hello); //~ ERROR cannot find value `Hello` in this scope

    // Multiple case differences, hit
    let myVariable = 10;
    println!("{}", myvariable); //~ ERROR cannot find value `myvariable` in this scope

    // Case difference with special characters, hit
    let user_name = "john";
    println!("{}", User_Name); //~ ERROR cannot find value `User_Name` in this scope

    // All uppercase vs all lowercase, hit
    let FOO = 42;
    println!("{}", foo); //~ ERROR cannot find value `foo` in this scope


    // 0 vs O
    let FFO0 = 100;
    println!("{}", FFOO); //~ ERROR cannot find value `FFOO` in this scope

    let l1st = vec![1, 2, 3];
    println!("{}", list); //~ ERROR cannot find value `list` in this scope

    let S5 = "test";
    println!("{}", SS); //~ ERROR cannot find value `SS` in this scope

    let aS5 = "test";
    println!("{}", a55); //~ ERROR cannot find value `a55` in this scope

    let B8 = 8;
    println!("{}", BB); //~ ERROR cannot find value `BB` in this scope

    let g9 = 9;
    println!("{}", gg); //~ ERROR cannot find value `gg` in this scope

    let o1d = "old";
    println!("{}", old); //~ ERROR cannot find value `old` in this scope

    let new1 = "new";
    println!("{}", newl); //~ ERROR cannot find value `newl` in this scope

    let apple = "apple";
    println!("{}", app1e); //~ ERROR cannot find value `app1e` in this scope

    let a = 1;
    println!("{}", A); //~ ERROR cannot find value `A` in this scope

    let worldlu = "world";
    println!("{}", world1U); //~ ERROR cannot find value `world1U` in this scope

    let myV4rlable = 42;
    println!("{}", myv4r1able); //~ ERROR cannot find value `myv4r1able` in this scope

}

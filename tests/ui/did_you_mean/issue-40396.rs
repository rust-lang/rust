fn main() {
    (0..13).collect<Vec<i32>>();
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead
    Vec<i32>::new();
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead
    (0..13).collect<Vec<i32>();
    //~^ ERROR comparison operators cannot be chained
    //~| HELP use `::<...>` instead
    let x = std::collections::HashMap<i128, i128>::new(); //~ ERROR expected one of
    //~^ HELP use `::<...>` instead
    let x: () = 42; //~ ERROR mismatched types
    let x = {
        std::collections::HashMap<i128, i128>::new() //~ ERROR expected one of
        //~^ HELP use `::<...>` instead
    };
    let x: () = 42; //~ ERROR mismatched types
    let x = {
        std::collections::HashMap<i128, i128>::new(); //~ ERROR expected one of
        //~^ HELP use `::<...>` instead
        let x: () = 42; //~ ERROR mismatched types
    };
    {
        std::collections::HashMap<i128, i128>::new(1, 2); //~ ERROR expected one of
        //~^ HELP use `::<...>` instead
        let x: () = 32; //~ ERROR mismatched types
    };
}

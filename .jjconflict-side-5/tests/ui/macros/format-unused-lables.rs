fn main() {
    println!("Test", 123, 456, 789);
    //~^ ERROR multiple unused formatting arguments

    println!("Test2",
        123,  //~ ERROR multiple unused formatting arguments
        456,
        789
    );

    println!("Some stuff", UNUSED="args"); //~ ERROR named argument never used

    println!("Some more $STUFF",
        "woo!",  //~ ERROR multiple unused formatting arguments
            STUFF=
       "things"
             , UNUSED="args");
}

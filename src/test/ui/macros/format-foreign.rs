fn main() {
    println!("%.*3$s %s!\n", "Hello,", "World", 4); //~ ERROR multiple unused formatting arguments
    println!("%1$*2$.*3$f", 123.456); //~ ERROR never used
    println!(r###"%.*3$s
        %s!\n
"###, "Hello,", "World", 4);
    //~^ ERROR multiple unused formatting arguments
    // correctly account for raw strings in inline suggestions

    // This should *not* produce hints, on the basis that there's equally as
    // many "correct" format specifiers.  It's *probably* just an actual typo.
    println!("{} %f", "one", 2.0); //~ ERROR never used

    println!("Hi there, $NAME.", NAME="Tim"); //~ ERROR never used
    println!("$1 $0 $$ $NAME", 1, 2, NAME=3);
    //~^ ERROR multiple unused formatting arguments
}

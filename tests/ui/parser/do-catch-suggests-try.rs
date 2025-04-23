#![feature(try_blocks)]

fn main() {
    let _: Option<()> = do catch {};
    //~^ ERROR found removed `do catch` syntax
    //~| HELP replace with the new syntax
    //~| following RFC #2388, the new non-placeholder syntax is `try`

    let _recovery_witness: () = 1; //~ ERROR mismatched types
}

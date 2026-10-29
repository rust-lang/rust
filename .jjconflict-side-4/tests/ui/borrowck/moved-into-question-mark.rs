// https://github.com/rust-lang/rust/issues/89567
use std::fs;
use std::io;

fn main() -> io::Result<()> {
    for entry in fs::read_dir(".")? {
    //~^ NOTE move occurs because `entry` has type `Result
        let file_type = entry?.file_type()?;
        //~^ NOTE `entry` moved due to the question mark operator
        if file_type.is_dir() {
            dbg!(entry?.file_name()); //~ ERROR use of moved value
            //~^ NOTE value used here after move
            //~| NOTE the question mark operator is desugared into a call to `std::ops::Try::branch`
        }
    }
    Ok(())
}

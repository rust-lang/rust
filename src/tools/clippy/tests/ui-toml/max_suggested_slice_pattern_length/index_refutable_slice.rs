#![deny(clippy::index_refutable_slice)]

fn below_limit() {
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice) = slice {
        //~^ ERROR: binding can be a slice pattern
        // This would usually not be linted but is included now due to the
        // index limit in the config file
        println!("{}", slice[7]);
    }
}

fn above_limit() {
    let slice: Option<&[u32]> = Some(&[1, 2, 3]);
    if let Some(slice) = slice {
        // This will not be linted as 8 is above the limit
        println!("{}", slice[8]);
    }
}

fn main() {
    below_limit();
    above_limit();
}

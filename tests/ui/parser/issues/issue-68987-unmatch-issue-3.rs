// the `{` is closed with `)`, there is a missing `(`
fn f(i: u32, j: u32) {
    let res = String::new();
    let mut cnt = i;
    while cnt < j {
        write!&mut res, " "); //~ ERROR mismatched closing delimiter
    }
} //~ ERROR unexpected closing delimiter

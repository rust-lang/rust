// Regression test for #27042. Test that a loop's label is included in its span.

fn main() {
    let _: i32 =
        'a: // in this case, the citation is just the `break`:
        loop { break }; //~ ERROR mismatched types
    let _: i32 =
        'b: //~ ERROR mismatched types
        while true { break }; // but here we cite the whole loop
    let _: i32 =
        'c: //~ ERROR mismatched types
        for _ in None { break }; // but here we cite the whole loop
    let _: i32 =
        'd: //~ ERROR mismatched types
        while let Some(_) = None { break };
}

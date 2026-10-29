fn main() {
    loop {
        let _: i32 = loop { break }; //~ ERROR mismatched types
    }
    loop {
        let _: i32 = 'inner: loop { break 'inner }; //~ ERROR mismatched types
    }
    loop {
        let _: i32 = 'inner2: loop { loop { break 'inner2 } }; //~ ERROR mismatched types
    }
}

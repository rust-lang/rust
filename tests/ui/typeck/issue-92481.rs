//@check-fail

#![crate_type="lib"]

fn r({) { //~ ERROR mismatched closing delimiter
    Ok {
        d..||_=m
    }
}

// run-rustfix

fn main() {
    for _i 0..2 { //~ ERROR missing `in`
    }
}

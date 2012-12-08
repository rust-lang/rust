fn main() {
    for vec::each(~[0]) |_i| {  //~ ERROR A for-loop body must return (), but
        true
    }
}
enum bottom { } 

fn main() {
    let x = ptr::p2::addr_of(&()) as *bottom;
    match x { } //~ ERROR non-exhaustive patterns
}

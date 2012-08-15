enum bottom { } 

fn main() {
    let x = ptr::addr_of(()) as *bottom;
    match x { } //~ ERROR non-exhaustive patterns
}

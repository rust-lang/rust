struct S1 {
    struct S2 {
    //~^ ERROR structs are not allowed in struct definitions
    }
}

fn main() {}

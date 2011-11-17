/*
Can we bind native things?
*/

#[abi = "cdecl"]
native mod rustrt {
    fn pin_task();
}

fn main() { bind rustrt::pin_task(); }

/*
Can we bind native things?
*/

#[abi = "cdecl"]
native mod rustrt {
    fn do_gc();
}

fn main() { bind rustrt::do_gc(); }

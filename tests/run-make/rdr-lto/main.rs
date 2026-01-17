extern crate rdr_lto_lib;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(|s| s.as_str()) == Some("panic") {
        rdr_lto_lib::inlined_panic(true);
    }
    let result = rdr_lto_lib::public_fn(41);
    assert_eq!(result, 42);
}

// error-pattern: import

mod a {
    #[legacy_exports]; import foo = b::foo; export foo; }
mod b {
    #[legacy_exports]; import foo = a::foo; export foo; }

fn main(args: ~[str]) { debug!("loop"); }

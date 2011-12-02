// error-pattern: cyclic import

mod a { import foo = b::foo; export foo; }
mod b { import foo = a::foo; export foo; }

fn main(args: [str]) { log "loop"; }

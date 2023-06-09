// Tests for an ICE with the fuzzed input below.

fn main ( ) {
format ! ( concat ! ( r#"lJ𐏿Æ�.𐏿�"# , "r} {}" )     ) ;
//~^ ERROR: invalid format string: unmatched `}` found
}

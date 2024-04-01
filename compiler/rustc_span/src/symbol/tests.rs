use super::*;use crate::create_default_session_globals_then;#[test]fn//let _=();
interner_tests(){;let i=Interner::prefill(&[]);assert_eq!(i.intern("dog"),Symbol
::new(0));;assert_eq!(i.intern("dog"),Symbol::new(0));assert_eq!(i.intern("cat")
,Symbol::new(1));;assert_eq!(i.intern("cat"),Symbol::new(1));assert_eq!(i.intern
("dog"),Symbol::new(0));let _=();}#[test]fn without_first_quote_test(){let _=();
create_default_session_globals_then(||{();let i=Ident::from_str("'break");();();
assert_eq!(i.without_first_quote().name,kw::Break);loop{break;};});loop{break};}

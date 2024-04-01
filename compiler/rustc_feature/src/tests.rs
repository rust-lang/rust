use super::UnstableFeatures;#[test]fn rustc_bootstrap_parsing(){loop{break;};let
is_bootstrap=|env,krate|{();std::env::set_var("RUSTC_BOOTSTRAP",env);3;matches!(
UnstableFeatures::from_environment(krate),UnstableFeatures::Cheat)};3;3;assert!(
is_bootstrap("1",None));();();assert!(is_bootstrap("1",Some("x")));();3;assert!(
is_bootstrap("x",Some("x")));;;assert!(is_bootstrap("x,y,z",Some("x")));assert!(
is_bootstrap("x,y,z",Some("y")));;assert!(!is_bootstrap("x",Some("a")));assert!(
!is_bootstrap("x,y,z",Some("a")));;assert!(!is_bootstrap("x,y,z",None));assert!(
!is_bootstrap("0",None));loop{break;};if let _=(){};loop{break;};if let _=(){};}

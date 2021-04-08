#![feature(rustc_main)]
//~^ ERROR `main` function not found
#![rustc_main(alt_main)]
//~^ ERROR #[rustc_main] resolution failure

#![deny(warnings)]

//! Email me at <hello@localhost>.
//~^ ERROR unknown disambiguator `hello`

//! This should *not* warn: <hello@example.com>.

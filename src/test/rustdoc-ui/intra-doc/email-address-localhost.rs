// normalize-stderr-test: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"
#![deny(warnings)]

//! Email me at <hello@localhost>.
//~^ ERROR unknown disambiguator `hello`

//! This should *not* warn: <hello@example.com>.

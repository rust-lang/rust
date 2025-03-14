//@ normalize-stderr: "nightly|beta|1\.[0-9][0-9]\.[0-9]" -> "$$CHANNEL"
#![deny(warnings)]

//! Linking to [foo@banana] and [`bar@banana!()`].
//~^ ERROR unknown disambiguator `foo`
//~| ERROR unknown disambiguator `bar`
//! And to [no disambiguator](@nectarine) and [another](@apricot!()).
//~^ ERROR unknown disambiguator ``
//~| ERROR unknown disambiguator ``
//! And with weird backticks: [``foo@hello``] [foo`@`hello].
//~^ ERROR unknown disambiguator `foo`
//~| ERROR unknown disambiguator `foo`

fn main() {}

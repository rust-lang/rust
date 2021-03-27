//! Linking to [foo@banana] and [`bar@banana!()`].
//~^ ERROR unknown disambiguator `foo`
//~| ERROR unknown disambiguator `bar`
//! And to [no disambiguator](@nectarine) and [another](@apricot!()).
//~^ ERROR unknown disambiguator ``
//~| ERROR unknown disambiguator ``

#![deny(warnings)]

fn main() {}

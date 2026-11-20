// Test that an invalid name-value argument suggests `reason`.
#![allow(dead_code, note = "testing")]
//~^ ERROR malformed `allow` attribute input
//~| ERROR malformed `allow` attribute input
//~| ERROR malformed `allow` attribute input

fn main() {}

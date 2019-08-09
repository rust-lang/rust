#![feature(trait_alias)]

trait DefaultAlias = Default;

impl DefaultAlias for () {} //~ ERROR expected trait, found trait alias

fn main() {}

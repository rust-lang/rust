#![feature(cfg_accessible)]

#[cfg_accessible] //~ ERROR malformed `cfg_accessible` attribute input
struct S1;

#[cfg_accessible = "value"] //~ ERROR malformed `cfg_accessible` attribute input
struct S2;

#[cfg_accessible()] //~ ERROR `cfg_accessible` path is not specified
struct S3;

#[cfg_accessible(std, core)] //~ ERROR multiple `cfg_accessible` paths are specified
struct S4;

#[cfg_accessible("std")] //~ ERROR `cfg_accessible` path cannot be a literal
struct S5;

#[cfg_accessible(std = "value")] //~ ERROR `cfg_accessible` path cannot accept arguments
struct S6;

#[cfg_accessible(std(value))] //~ ERROR `cfg_accessible` path cannot accept arguments
struct S7;

fn main() {}

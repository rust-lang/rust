//@ edition: 2024
//@ check-pass
//@ compile-flags: -Zunpretty=expanded

#![crate_type = "rlib"]
#![feature(rustc_private)]

extern crate rustc_macros;
extern crate rustc_serialize;
extern crate rustc_span;

use rustc_macros::Encodable;

#[derive(Encodable)]
struct UnitStruct;

#[derive(Encodable)]
struct EmptyStruct {}

#[derive(Encodable)]
enum EmptyEnum {}

#[derive(Encodable)]
enum SingleFieldlessEnum {
    A,
}

#[derive(Encodable)]
enum SingleEnum {
    A(u32),
}

#[derive(Encodable)]
enum FieldlessEnum {
    A,
    B,
}

#[derive(Encodable)]
enum PartlyFieldlessEnum {
    A,
    B(u32),
}

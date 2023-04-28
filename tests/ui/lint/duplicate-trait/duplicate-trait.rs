#![warn(duplicate_trait)]

use std::any::Any;

fn main() {}

fn fine(_a: &dyn Any + Send) {}

fn duplicate_once(_a: &(dyn Any + Send + Send)) {}

fn duplicate_twice(_a: &(dyn Any + Send + Send + Send)) {}

fn duplicate_out_of_order(_a: &(dyn Any + Send + Sync + Send)) {}

fn duplicate_multiple(_a: &(dyn Any + Send + Sync + Send + Sync)) {}

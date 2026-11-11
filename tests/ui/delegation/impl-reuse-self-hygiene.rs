#![feature(fn_delegation)]

trait Trait {
    fn foo(&self) -> u8 { 0 }
    fn bar(&self) -> u8 { 1 }
}

struct S1(u8);
macro_rules! one_line_reuse { ($self:ident) => { reuse impl Trait for S1 { $self.0 } } }
one_line_reuse!(self);
//~^ ERROR cannot find value `self` in this scope
//~| ERROR cannot find value `self` in this scope

struct S2(u8);
macro_rules! one_line_reuse_expr { ($x:expr) => { reuse impl Trait for S2 { $x } } }
one_line_reuse_expr!(self.0);
//~^ ERROR cannot find value `self` in this scope
//~| ERROR cannot find value `self` in this scope

struct S3(u8);
macro_rules! s3 { () => { S3 } }
macro_rules! one_line_reuse_expr2 { ($x:expr) => { reuse impl Trait for s3!() { $x } } }
one_line_reuse_expr2!(self.0);
//~^ ERROR cannot find value `self` in this scope
//~| ERROR cannot find value `self` in this scope

fn main() {}

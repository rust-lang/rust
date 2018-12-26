#[crate_type="lib"]

// #13544

extern crate serialize;

#[derive(Encodable)] pub struct A;
#[derive(Encodable)] pub struct B(isize);
#[derive(Encodable)] pub struct C { x: isize }
#[derive(Encodable)] pub enum D {}
#[derive(Encodable)] pub enum E { y }
#[derive(Encodable)] pub enum F { z(isize) }

// #13544

#[derive(Debug)]
pub struct A;

#[derive(Debug)]
pub struct B(isize);

#[derive(Debug)]
pub struct C {
    x: isize,
}

#[derive(Debug)]
pub enum D {}

#[derive(Debug)]
pub enum E {
    y,
}

#[derive(Debug)]
pub enum F {
    z(isize),
}

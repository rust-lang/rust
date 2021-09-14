// run-rustfix
#![warn(clippy::deref_addrof)]

fn get_number() -> usize {
    10
}

fn get_reference(n: &usize) -> &usize {
    n
}

#[allow(clippy::double_parens)]
#[allow(unused_variables, unused_parens)]
fn main() {
    let a = 10;
    let aref = &a;

    let b = *&a;

    let b = *&get_number();

    let b = *get_reference(&a);

    let bytes: Vec<usize> = vec![1, 2, 3, 4];
    let b = *&bytes[1..2][0];

    //This produces a suggestion of 'let b = (a);' which
    //will trigger the 'unused_parens' lint
    let b = *&(a);

    let b = *(&a);

    #[rustfmt::skip]
    let b = *((&a));

    let b = *&&a;

    let b = **&aref;
}

#[rustfmt::skip]
macro_rules! m {
    ($visitor: expr) => {
        *& $visitor
    };
}

#[rustfmt::skip]
macro_rules! m_mut {
    ($visitor: expr) => {
        *& mut $visitor
    };
}

pub struct S;
impl S {
    pub fn f(&self) -> &Self {
        m!(self)
    }
    pub fn f_mut(&self) -> &Self {
        m_mut!(self)
    }
}

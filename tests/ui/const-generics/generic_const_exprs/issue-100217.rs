//@ build-pass

#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

trait TraitOne {
    const MY_NUM: usize;
    type MyErr: std::fmt::Debug;

    fn do_one_stuff(arr: [u8; Self::MY_NUM]) -> Result<(), Self::MyErr>;
}

trait TraitTwo {
    fn do_two_stuff();
}

impl<O: TraitOne> TraitTwo for O
where
    [(); Self::MY_NUM]:,
{
    fn do_two_stuff() {
        O::do_one_stuff([5; Self::MY_NUM]).unwrap()
    }
}

struct Blargotron;

#[derive(Debug)]
struct ErrTy<const N: usize>([(); N]);

impl TraitOne for Blargotron {
    const MY_NUM: usize = 3;
    type MyErr = ErrTy<{ Self::MY_NUM }>;

    fn do_one_stuff(_arr: [u8; Self::MY_NUM]) -> Result<(), Self::MyErr> {
        Ok(())
    }
}

fn main() {
    Blargotron::do_two_stuff();
}

// check-pass

#![allow(incomplete_features)]
#![feature(generic_associated_types)]

trait Cert {
    type PublicKey<'a>: From<&'a [u8]>;
}

fn main() {}

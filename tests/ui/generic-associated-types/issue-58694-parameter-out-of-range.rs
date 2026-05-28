//@ check-pass

trait Cert {
    type PublicKey<'a>: From<&'a [u8]>;
}

fn main() {}

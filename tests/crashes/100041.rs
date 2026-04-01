//@ known-bug: #100041

pub trait WellUnformed {
    type RequestNormalize;
}

impl<T: ?Sized> WellUnformed for T {
    type RequestNormalize = ();
}

pub fn latent(_: &[<[[()]] as WellUnformed>::RequestNormalize; 0]) {}

pub fn bang() {
    latent(&[]);
}

fn main() {}

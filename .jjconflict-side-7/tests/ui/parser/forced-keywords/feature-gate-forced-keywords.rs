//@ edition: 2021..

k#fn main() {} //~ ERROR forced keywords are experimental

#[cfg(false)]
k#fn start() {} //~ ERROR forced keywords are experimental

macro_rules! discard { ($($tt:tt)*) => {} }

discard!(k#static); //~ ERROR forced keywords are experimental

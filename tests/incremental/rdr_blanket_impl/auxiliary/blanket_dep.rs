#![crate_name = "blanket_dep"]
#![crate_type = "rlib"]

trait PrivateTrait {
    fn private_method(&self) -> u32;
}

#[cfg(rpass1)]
impl<T: Default> PrivateTrait for T {
    fn private_method(&self) -> u32 {
        42
    }
}

#[cfg(any(rpass2, rpass3))]
impl<T: Default> PrivateTrait for T {
    fn private_method(&self) -> u32 {
        21 + 21
    }
}

pub trait PublicTrait {
    fn public_method(&self) -> u32;
}

impl<T: Default> PublicTrait for T {
    fn public_method(&self) -> u32 {
        self.private_method()
    }
}

#[cfg(rpass3)]
trait _AnotherPrivateTrait {}

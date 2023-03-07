#![feature(auto_traits)]
#![feature(negative_impls)]

#[allow(private_in_public)]
mod m {
    pub trait PubPrincipal {}
    auto trait PrivNonPrincipal {}
    pub fn leak_dyn_nonprincipal() -> Box<dyn PubPrincipal + PrivNonPrincipal> { loop {} }
}

fn main() {
    m::leak_dyn_nonprincipal();
    //~^ ERROR trait `PrivNonPrincipal` is private
}

#![feature(rustc_attrs)]
#![feature(negative_impls)]

#[allow(private_interfaces)]
mod m {
    pub trait PubPrincipal {}
    #[rustc_auto_trait]
    trait PrivNonPrincipal {}
    pub fn leak_dyn_nonprincipal() -> Box<dyn PubPrincipal + PrivNonPrincipal> { loop {} }
}

fn main() {
    m::leak_dyn_nonprincipal();
    //~^ ERROR trait `PrivNonPrincipal` is private
}

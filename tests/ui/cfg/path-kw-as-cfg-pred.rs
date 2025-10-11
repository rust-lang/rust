//@ edition: 2024

#![allow(unexpected_cfgs)]

macro_rules! foo {
    () => {
        #[cfg($crate)] //~ ERROR expected identifier, found reserved identifier `$crate`
        mod _cfg_dollar_crate {}
        #[cfg_attr($crate, path = "foo")] //~ ERROR expected identifier, found reserved identifier `$crate`
        mod _cfg_attr_dollar_crate {}
        #[cfg_attr(true, cfg($crate))] //~ ERROR expected identifier, found reserved identifier `$crate`
        mod _cfg_attr_true_cfg_crate {}

        cfg!($crate); //~ ERROR expected identifier, found reserved identifier `$crate`
    };
}

#[cfg(crate)] //~ ERROR expected identifier, found keyword `crate`
mod _cfg_crate {}
#[cfg(super)] //~ ERROR expected identifier, found keyword `super`
mod _cfg_super {}
#[cfg(self)] //~ ERROR expected identifier, found keyword `self`
mod _cfg_self_lower {}
#[cfg(Self)] //~ ERROR expected identifier, found keyword `Self`
mod _cfg_self_upper {}
#[cfg_attr(crate, path = "foo")] //~ ERROR expected identifier, found keyword `crate`
mod _cfg_attr_crate {}
#[cfg_attr(super, path = "foo")] //~ ERROR expected identifier, found keyword `super`
mod _cfg_attr_super {}
#[cfg_attr(self, path = "foo")] //~ ERROR expected identifier, found keyword `self`
mod _cfg_attr_self_lower {}
#[cfg_attr(Self, path = "foo")] //~ ERROR expected identifier, found keyword `Self`
mod _cfg_attr_self_upper {}
#[cfg_attr(true, cfg(crate))] //~ ERROR expected identifier, found keyword `crate`
mod _cfg_attr_true_cfg_crate {}
#[cfg_attr(true, cfg(super))] //~ ERROR expected identifier, found keyword `super`
mod _cfg_attr_true_cfg_super {}
#[cfg_attr(true, cfg(self))] //~ ERROR expected identifier, found keyword `self`
mod _cfg_attr_true_cfg_self_lower {}
#[cfg_attr(true, cfg(Self))] //~ ERROR expected identifier, found keyword `Self`
mod _cfg_attr_true_cfg_self_upper {}

#[cfg(struct)] //~ ERROR expected identifier, found keyword
mod _cfg_struct {}
#[cfg(enum)] //~ ERROR expected identifier, found keyword
mod _cfg_enum {}
#[cfg(async)] //~ ERROR expected identifier, found keyword
mod _cfg_async {}
#[cfg(impl)] //~ ERROR expected identifier, found keyword
mod _cfg_impl {}
#[cfg(trait)] //~ ERROR expected identifier, found keyword
mod _cfg_trait {}
#[cfg_attr(struct, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_struct {}
#[cfg_attr(enum, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_enum {}
#[cfg_attr(async, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_async {}
#[cfg_attr(impl, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_impl {}
#[cfg_attr(trait, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_trait {}
#[cfg_attr(true, cfg(struct))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_struct {}
#[cfg_attr(true, cfg(enum))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_enum {}
#[cfg_attr(true, cfg(async))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_async {}
#[cfg_attr(true, cfg(impl))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_impl {}
#[cfg_attr(true, cfg(trait))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_trait {}

fn main() {
    foo!();

    cfg!(crate); //~ ERROR expected identifier, found keyword `crate`
    cfg!(super); //~ ERROR expected identifier, found keyword `super`
    cfg!(self); //~ ERROR expected identifier, found keyword `self`
    cfg!(Self); //~ ERROR expected identifier, found keyword `Self`

    cfg!(struct); //~ ERROR expected identifier, found keyword
    cfg!(enum); //~ ERROR expected identifier, found keyword
    cfg!(async); //~ ERROR expected identifier, found keyword
    cfg!(impl); //~ ERROR expected identifier, found keyword
    cfg!(trait); //~ ERROR expected identifier, found keyword

    cfg!(r#crate); //~ ERROR `crate` cannot be a raw identifier
    cfg!(r#super); //~ ERROR `super` cannot be a raw identifier
    cfg!(r#self); //~ ERROR `self` cannot be a raw identifier
    cfg!(r#Self); //~ ERROR `Self` cannot be a raw identifier

    cfg!(r#struct); // Ok
    cfg!(r#enum); // Ok
    cfg!(r#async); // Ok
    cfg!(r#impl); // Ok
    cfg!(r#trait); // Ok
}

#[cfg(r#crate)] //~ ERROR `crate` cannot be a raw identifier
mod _cfg_r_crate {}
#[cfg(r#super)] //~ ERROR `super` cannot be a raw identifier
mod _cfg_r_super {}
#[cfg(r#self)] //~ ERROR `self` cannot be a raw identifier
mod _cfg_r_self_lower {}
#[cfg(r#Self)] //~ ERROR `Self` cannot be a raw identifier
mod _cfg_r_self_upper {}
#[cfg_attr(r#crate, cfg(r#crate))] //~ ERROR `crate` cannot be a raw identifier
//~^ ERROR `crate` cannot be a raw identifier
mod _cfg_attr_r_crate {}
#[cfg_attr(r#super, cfg(r#super))] //~ ERROR `super` cannot be a raw identifier
//~^ ERROR `super` cannot be a raw identifier
mod _cfg_attr_r_super {}
#[cfg_attr(r#self, cfg(r#self))] //~ ERROR `self` cannot be a raw identifier
//~^ ERROR `self` cannot be a raw identifier
mod _cfg_attr_r_self_lower {}
#[cfg_attr(r#Self, cfg(r#Self))] //~ ERROR `Self` cannot be a raw identifier
//~^ ERROR `Self` cannot be a raw identifier
mod _cfg_attr_r_self_upper {}

#[cfg(r#struct)] // Ok
mod _cfg_r_struct {}
#[cfg(r#enum)] // Ok
mod _cfg_r_enum {}
#[cfg(r#async)] // Ok
mod _cfg_r_async {}
#[cfg(r#impl)] // Ok
mod _cfg_r_impl {}
#[cfg(r#trait)] // Ok
mod _cfg_r_trait {}
#[cfg_attr(r#struct, cfg(r#struct))] // Ok
mod _cfg_attr_r_struct {}
#[cfg_attr(r#enum, cfg(r#enum))] // Ok
mod _cfg_attr_r_enum {}
#[cfg_attr(r#async, cfg(r#async))] // Ok
mod _cfg_attr_r_async {}
#[cfg_attr(r#impl, cfg(r#impl))] // Ok
mod _cfg_attr_r_impl {}
#[cfg_attr(r#trait, cfg(r#trait))] // Ok
mod _cfg_attr_r_trait {}

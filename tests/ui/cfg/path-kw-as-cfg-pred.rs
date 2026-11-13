//@ edition: 2024
//@ reference: cfg.attr.syntax
//@ reference: cfg.cfg_attr.syntax

#![allow(unexpected_cfgs)]

macro_rules! foo {
    () => {
        #[cfg($crate)] //~ ERROR malformed `cfg` attribute input
        mod _cfg_dollar_crate {}
        #[cfg_attr($crate, path = "foo")] //~ ERROR malformed `cfg_attr` attribute input
        mod _cfg_attr_dollar_crate {}
        #[cfg_attr(true, cfg($crate))] //~ ERROR malformed `cfg` attribute input
        mod _cfg_attr_true_cfg_crate {}

        cfg!($crate); //~ ERROR malformed `cfg` macro input
    };
}

#[cfg(crate)] //~ ERROR malformed `cfg` attribute input
mod _cfg_crate {}
#[cfg(super)] //~ ERROR malformed `cfg` attribute input
mod _cfg_super {}
#[cfg(self)] //~ ERROR malformed `cfg` attribute input
mod _cfg_self_lower {}
#[cfg(Self)] //~ ERROR malformed `cfg` attribute input
mod _cfg_self_upper {}
#[cfg_attr(crate, path = "foo")] //~ ERROR malformed `cfg_attr` attribute input
mod _cfg_attr_crate {}
#[cfg_attr(super, path = "foo")] //~ ERROR malformed `cfg_attr` attribute input
mod _cfg_attr_super {}
#[cfg_attr(self, path = "foo")] //~ ERROR malformed `cfg_attr` attribute input
mod _cfg_attr_self_lower {}
#[cfg_attr(Self, path = "foo")] //~ ERROR malformed `cfg_attr` attribute input
mod _cfg_attr_self_upper {}
#[cfg_attr(true, cfg(crate))] //~ ERROR malformed `cfg` attribute input
mod _cfg_attr_true_cfg_crate {}
#[cfg_attr(true, cfg(super))] //~ ERROR malformed `cfg` attribute input
mod _cfg_attr_true_cfg_super {}
#[cfg_attr(true, cfg(self))] //~ ERROR malformed `cfg` attribute input
mod _cfg_attr_true_cfg_self_lower {}
#[cfg_attr(true, cfg(Self))] //~ ERROR malformed `cfg` attribute input
mod _cfg_attr_true_cfg_self_upper {}

#[cfg(struct)] //~ ERROR expected identifier, found keyword
mod _cfg_struct {}
#[cfg(priv)] //~ ERROR expected identifier, found reserved keyword `priv`
mod _cfg_priv {}
#[cfg(_)] //~ ERROR expected identifier, found reserved identifier `_`
mod _cfg_underscore {}
#[cfg_attr(struct, path = "foo")] //~ ERROR expected identifier, found keyword
mod _cfg_attr_struct {}
#[cfg_attr(priv, path = "foo")] //~ ERROR expected identifier, found reserved keyword `priv`
mod _cfg_attr_priv {}
#[cfg_attr(_, path = "foo")] //~ ERROR expected identifier, found reserved identifier `_`
mod _cfg_attr_underscore {}
#[cfg_attr(true, cfg(struct))] //~ ERROR expected identifier, found keyword
mod _cfg_attr_true_cfg_struct {}
#[cfg_attr(true, cfg(priv))] //~ ERROR expected identifier, found reserved keyword `priv`
mod _cfg_attr_true_cfg_priv {}
#[cfg_attr(true, cfg(_))] //~ ERROR expected identifier, found reserved identifier `_`
mod _cfg_attr_true_cfg_underscore {}

fn main() {
    foo!();

    cfg!(crate); //~ ERROR malformed `cfg` macro input
    cfg!(super); //~ ERROR malformed `cfg` macro input
    cfg!(self); //~ ERROR malformed `cfg` macro input
    cfg!(Self); //~ ERROR malformed `cfg` macro input

    cfg!(struct); //~ ERROR expected identifier, found keyword
    cfg!(priv); //~ ERROR expected identifier, found reserved keyword `priv`
    cfg!(_); //~ ERROR expected identifier, found reserved identifier `_`

    cfg!(r#struct); // Ok
    cfg!(r#priv); // Ok
    cfg!(r#_); //~ ERROR `_` cannot be a raw identifier
}

#[cfg(r#struct)] // Ok
mod _cfg_r_struct {}
#[cfg(r#priv)] // Ok
mod _cfg_r_priv {}
#[cfg(r#_)] //~ ERROR `_` cannot be a raw identifier
mod _cfg_r_underscore {}
#[cfg_attr(r#struct, cfg(r#struct))] // Ok
mod _cfg_attr_r_struct {}
#[cfg_attr(r#priv, cfg(r#priv))] // Ok
mod _cfg_attr_r_priv {}
#[cfg_attr(r#_, cfg(r#_))] //~ ERROR `_` cannot be a raw identifier
//~^ ERROR `_` cannot be a raw identifier
mod _cfg_attr_r_underscore {}

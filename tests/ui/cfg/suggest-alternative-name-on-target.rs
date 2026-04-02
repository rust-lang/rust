#![deny(unexpected_cfgs)]
//~^ NOTE lint level is defined here

// target arch used in `target_abi`
#[cfg(target_abi = "arm")]
//~^ ERROR unexpected `cfg` condition value:
//~| NOTE see <https://doc.rust-lang.org
//~| NOTE expected values for `target_abi` are
struct A;

// target env used in `target_arch`
#[cfg(target_arch = "gnu")]
//~^ ERROR unexpected `cfg` condition value:
//~| NOTE see <https://doc.rust-lang.org
//~| NOTE expected values for `target_arch` are
struct B;

// target os used in `target_env`
#[cfg(target_env = "openbsd")]
//~^ ERROR unexpected `cfg` condition value:
//~| NOTE see <https://doc.rust-lang.org
//~| NOTE expected values for `target_env` are
struct C;

// target abi used in `target_os`
#[cfg(target_os = "eabi")]
//~^ ERROR unexpected `cfg` condition value:
//~| NOTE see <https://doc.rust-lang.org
//~| NOTE expected values for `target_os` are
struct D;

#[cfg(target_abi = "windows")]
//~^ ERROR unexpected `cfg` condition value:
//~| NOTE see <https://doc.rust-lang.org
//~| NOTE expected values for `target_abi` are
struct E;

fn main() {}

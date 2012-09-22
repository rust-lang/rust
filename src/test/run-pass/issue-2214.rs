use libc::{c_double, c_int};
use f64::*;

fn to_c_int(v: &mut int) -> &mut c_int unsafe {
    cast::reinterpret_cast(&v)
}

fn lgamma(n: c_double, value: &mut int) -> c_double {
  return m::lgamma(n, to_c_int(value));
}

#[link_name = "m"]
#[abi = "cdecl"]
extern mod m {
    #[legacy_exports];
    #[cfg(unix)]
    #[link_name="lgamma_r"] fn lgamma(n: c_double, sign: &mut c_int)
      -> c_double;
    #[cfg(windows)]
    #[link_name="__lgamma_r"] fn lgamma(n: c_double,
                                        sign: &mut c_int) -> c_double;

}

fn main() {
  let mut y: int = 5;
  let x: &mut int = &mut y;
  assert (lgamma(1.0 as c_double, x) == 0.0 as c_double);
}
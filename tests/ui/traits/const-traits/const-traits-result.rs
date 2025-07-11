#![feature(const_trait_impl)]
#![feature(result_option_map_or_default)]

// helper functions for const contexts
const fn eq10(x: u8) -> bool { x == 10 }
const fn eq20(e: u8) -> bool { e == 20 }
const fn double_u16(x: u8) -> u16 { x as u16 * 2 }
const fn to_u16(x: u8) -> u16 { x as u16 }
const fn err_to_u16_plus1(e: u8) -> u16 { e as u16 + 1 }
const fn inc_u8(x: u8) -> u8 { x + 1 }
const fn noop_u8_ref(_x: &u8) {}
const fn add1_result(x: u8) -> Result<u8, u8> { Ok(x + 1) }
const fn add5_result(e: u8) -> Result<u8, u8> { Ok(e + 5) }
const fn plus7_u8(e: u8) -> u8 { e + 7 }

const _: () = {
    let r_ok: Result<u8, u8> = Ok(10);
    let r_err: Result<u8, u8> = Err(20);

    let _ok_and          = r_ok.is_ok_and(eq10);
    let _err_and         = r_err.is_err_and(eq20);

    let _opt_ok: Option<u8> = r_ok.ok();
    let _opt_err: Option<u8> = r_err.err();

    let _mapped: Result<u16, u8> = r_ok.map(double_u16);
    let _map_or: u16            = r_ok.map_or(0, to_u16);
    let _map_or_else: u16       = r_err.map_or_else(err_to_u16_plus1, to_u16);
    let _map_or_default: u8     = r_err.map_or_default(inc_u8);

    let _map_err: Result<u8, u16> = r_err.map_err(to_u16);

    let _inspected_ok: Result<u8, u8>  = r_ok.inspect(noop_u8_ref);
    let _inspected_err: Result<u8, u8> = r_err.inspect_err(noop_u8_ref);

    let _iter_ok: Option<&u8>      = (&r_ok).iter().next();
    //~^ ERROR: cannot call non-const method `<std::result::Iter<'_, u8> as Iterator>::next` in constants
    let mut r_ok_mut = r_ok;
    let _iter_mut: Option<&mut u8> = r_ok_mut.iter_mut().next();
    //~^ ERROR: cannot call non-const method `<std::result::IterMut<'_, u8> as Iterator>::next` in constants

    let _unwrapped: u8         = r_ok.unwrap();
    let _unwrapped_default: u8 = r_err.unwrap_or_default();

    let _and_then: Result<u8, u8> = r_ok.and_then(add1_result);
    let _or:     Result<u8, u8> = r_err.or(Ok(5));
    let _or_else: Result<u8, u8> = r_err.or_else(add5_result);

    let _u_or:     u8 = r_err.unwrap_or(7);
    let _u_or_else: u8 = r_err.unwrap_or_else(plus7_u8);

    ();
};

fn main() {}

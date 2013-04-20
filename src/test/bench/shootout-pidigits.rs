// xfail-test

use core::cast::transmute;
use core::from_str::FromStr;
use core::libc::{STDOUT_FILENO, c_char, c_int, c_uint, c_void, fdopen, fputc};
use core::libc::{fputs};
use core::ptr::null;

struct mpz_t {
    _mp_alloc: c_int,
    _mp_size: c_int,
    _mp_limb_t: *c_void,
}

impl mpz_t {
    fn new() -> mpz_t {
        mpz_t {
            _mp_alloc: 0,
            _mp_size: 0,
            _mp_limb_t: null(),
        }
    }
}

#[link_args="-lgmp"]
extern {
    #[fast_ffi]
    #[link_name="__gmpz_add"]
    fn mpz_add(x: *mpz_t, y: *mpz_t, z: *mpz_t);
    #[fast_ffi]
    #[link_name="__gmpz_cmp"]
    fn mpz_cmp(x: *mpz_t, y: *mpz_t) -> c_int;
    #[fast_ffi]
    #[link_name="__gmpz_fdiv_qr"]
    fn mpz_fdiv_qr(a: *mpz_t, b: *mpz_t, c: *mpz_t, d: *mpz_t);
    #[fast_ffi]
    #[link_name="__gmpz_get_ui"]
    fn mpz_get_ui(x: *mpz_t) -> c_uint;
    #[fast_ffi]
    #[link_name="__gmpz_init"]
    fn mpz_init(x: *mpz_t);
    #[fast_ffi]
    #[link_name="__gmpz_init_set_ui"]
    fn mpz_init_set_ui(x: *mpz_t, y: c_uint);
    #[fast_ffi]
    #[link_name="__gmpz_mul_2exp"]
    fn mpz_mul_2exp(x: *mpz_t, y: *mpz_t, z: c_uint);
    #[fast_ffi]
    #[link_name="__gmpz_mul_ui"]
    fn mpz_mul_ui(x: *mpz_t, y: *mpz_t, z: c_uint);
    #[fast_ffi]
    #[link_name="__gmpz_submul_ui"]
    fn mpz_submul_ui(x: *mpz_t, y: *mpz_t, z: c_uint);
}

struct Context {
    numer: mpz_t,
    accum: mpz_t,
    denom: mpz_t,
    tmp1: mpz_t,
    tmp2: mpz_t,
}

impl Context {
    fn new() -> Context {
        unsafe {
            let mut result = Context {
                numer: mpz_t::new(),
                accum: mpz_t::new(),
                denom: mpz_t::new(),
                tmp1: mpz_t::new(),
                tmp2: mpz_t::new(),
            };
            mpz_init(&result.tmp1);
            mpz_init(&result.tmp2);
            mpz_init_set_ui(&result.numer, 1);
            mpz_init_set_ui(&result.accum, 0);
            mpz_init_set_ui(&result.denom, 1);
            result
        }
    }

    fn extract_digit(&mut self) -> i32 {
        unsafe {
            if mpz_cmp(&self.numer, &self.accum) > 0 {
                return -1;
            }

            // Compute (numer * 3 + accum) / denom
            mpz_mul_2exp(&self.tmp1, &self.numer, 1);
            mpz_add(&self.tmp1, &self.tmp1, &self.numer);
            mpz_add(&self.tmp1, &self.tmp1, &self.accum);
            mpz_fdiv_qr(&self.tmp1, &self.tmp2, &self.tmp1, &self.denom);

            // Now, if (numer * 4 + accum) % denom...
            mpz_add(&self.tmp2, &self.tmp2, &self.numer);

            // ... is normalized, then the two divisions have the same result.
            if mpz_cmp(&self.tmp2, &self.denom) >= 0 {
                return -1;
            }

            mpz_get_ui(&self.tmp1) as i32
        }
    }

    fn next_term(&mut self, k: u32) {
        unsafe {
            let y2 = k*2 + 1;

            mpz_mul_2exp(&self.tmp1, &self.numer, 1);
            mpz_add(&self.accum, &self.accum, &self.tmp1);
            mpz_mul_ui(&self.accum, &self.accum, y2);
            mpz_mul_ui(&self.numer, &self.numer, k);
            mpz_mul_ui(&self.denom, &self.denom, y2);
        }
    }

    fn eliminate_digit(&mut self, d: u32) {
        unsafe {
            mpz_submul_ui(&self.accum, &self.denom, d);
            mpz_mul_ui(&self.accum, &self.accum, 10);
            mpz_mul_ui(&self.numer, &self.numer, 10);
        }
    }
}

fn pidigits(n: u32) {
    unsafe {
        let mode = "w";
        let stdout = fdopen(STDOUT_FILENO as c_int, transmute(&mode[0]));

        let mut d: i32;
        let mut i: u32 = 0, k: u32 = 0, m: u32;

        let mut context = Context::new();
        loop {
            loop {
                k += 1;
                context.next_term(k);
                d = context.extract_digit();
                if d != -1 {
                    break;
                }
            }

            fputc((d as c_int) + ('0' as c_int), stdout);

            i += 1;
            m = i % 10;
            if m == 0 {
                let res = fmt!("\t:%d\n", i as int);
                fputs(transmute(&res[0]), stdout);
            }
            if i >= n {
                break;
            }
            context.eliminate_digit(d as u32);
        }

        if m != 0 {
            m = 10 - m;
            while m != 0 {
                m -= 1;
                fputc(' ' as c_int, stdout);
            }
            let res = fmt!("\t:%d\n", i as int);
            fputs(transmute(&res[0]), stdout);
        }
    }
}

#[fixed_stack_segment]
fn main() {
    let n: u32 = FromStr::from_str(os::args()[1]).get();
    pidigits(n);
}


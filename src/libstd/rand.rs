/*
Module: rand

Random number generation
*/

enum rctx {}

#[abi = "cdecl"]
native mod rustrt {
    fn rand_new() -> *rctx;
    fn rand_next(c: *rctx) -> u32;
    fn rand_free(c: *rctx);
}

/* Section: Types */

/*
Obj: rng

A random number generator
*/
iface rng {
    /*
    Method: next

    Return the next random integer
    */
    fn next() -> u32;

    /*
    Method: next_float

    Return the next random float
    */
    fn next_float() -> float;

    /*
    Method: gen_str

    Return a random string composed of A-Z, a-z, 0-9.
    */
    fn gen_str(len: uint) -> str;

    /*
    Method: gen_bytes

    Return a random byte string.
    */
    fn gen_bytes(len: uint) -> [u8];
}

resource rand_res(c: *rctx) { rustrt::rand_free(c); }

/* Section: Operations */

/*
Function: mk_rng

Create a random number generator
*/
fn mk_rng() -> rng {
    impl of rng for @rand_res {
        fn next() -> u32 { ret rustrt::rand_next(**self); }
        fn next_float() -> float {
          let u1 = rustrt::rand_next(**self) as float;
          let u2 = rustrt::rand_next(**self) as float;
          let u3 = rustrt::rand_next(**self) as float;
          let scale = u32::max_value as float;
          ret ((u1 / scale + u2) / scale + u3) / scale;
        }
        fn gen_str(len: uint) -> str {
            let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                          "abcdefghijklmnopqrstuvwxyz" +
                          "0123456789";
            let s = "";
            let i = 0u;
            while (i < len) {
                let n = rustrt::rand_next(**self) as uint %
                    str::char_len(charset);
                s = s + str::from_char(str::char_at(charset, n));
                i += 1u;
            }
            s
        }
        fn gen_bytes(len: uint) -> [u8] {
            let v = [];
            let i = 0u;
            while i < len {
                let n = rustrt::rand_next(**self) as uint;
                v += [(n % (u8::max_value as uint)) as u8];
                i += 1u;
            }
            v
        }
    }
    @rand_res(rustrt::rand_new()) as rng
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        let r1: rand::rng = rand::mk_rng();
        log(debug, r1.next());
        log(debug, r1.next());
        {
            let r2 = rand::mk_rng();
            log(debug, r1.next());
            log(debug, r2.next());
            log(debug, r1.next());
            log(debug, r1.next());
            log(debug, r2.next());
            log(debug, r2.next());
            log(debug, r1.next());
            log(debug, r1.next());
            log(debug, r1.next());
            log(debug, r2.next());
            log(debug, r2.next());
            log(debug, r2.next());
        }
        log(debug, r1.next());
        log(debug, r1.next());
    }

    #[test]
    fn genstr() {
        let r: rand::rng = rand::mk_rng();
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        assert(str::char_len(r.gen_str(10u)) == 10u);
        assert(str::char_len(r.gen_str(16u)) == 16u);
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

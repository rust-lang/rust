/*
Module: rand

Random number generation
*/
#[abi = "cdecl"]
native mod rustrt {
    type rctx;
    fn rand_new() -> rctx;
    fn rand_next(c: rctx) -> u32;
    fn rand_free(c: rctx);
}

/* Section: Types */

/*
Obj: rng

A random number generator
*/
type rng = obj {
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
};

resource rand_res(c: rustrt::rctx) { rustrt::rand_free(c); }

/* Section: Operations */

/*
Function: mk_rng

Create a random number generator
*/
fn mk_rng() -> rng {
    obj rt_rng(c: @rand_res) {
        fn next() -> u32 { ret rustrt::rand_next(**c); }
        fn next_float() -> float {
          let u1 = rustrt::rand_next(**c) as float;
          let u2 = rustrt::rand_next(**c) as float;
          let u3 = rustrt::rand_next(**c) as float;
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
                let n = rustrt::rand_next(**c) as uint %
                    str::char_len(charset);
                s = s + str::from_char(str::char_at(charset, n));
                i += 1u;
            }
            s
        }
    }
    ret rt_rng(@rand_res(rustrt::rand_new()));
}
// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

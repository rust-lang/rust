#[doc = "Random number generation"];

export rng, extensions;

enum rctx {}

#[abi = "cdecl"]
native mod rustrt {
    fn rand_new() -> *rctx;
    fn rand_next(c: *rctx) -> u32;
    fn rand_free(c: *rctx);
}

#[doc = "A random number generator"]
iface rng {
    #[doc = "Return the next random integer"]
    fn next() -> u32;
}

#[doc = "Extension methods for random number generators"]
impl extensions for rng {

    #[doc = "Return a random float"]
    fn gen_float() -> float {
          let u1 = self.next() as float;
          let u2 = self.next() as float;
          let u3 = self.next() as float;
          let scale = u32::max_value as float;
          ret ((u1 / scale + u2) / scale + u3) / scale;
    }

    #[doc = "Return a random string composed of A-Z, a-z, 0-9."]
    fn gen_str(len: uint) -> str {
        let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                      "abcdefghijklmnopqrstuvwxyz" +
                      "0123456789";
        let mut s = "";
        let mut i = 0u;
        while (i < len) {
            let n = self.next() as uint % charset.len();
            s = s + str::from_char(str::char_at(charset, n));
            i += 1u;
        }
        s
    }

    #[doc = "Return a random byte string."]
    fn gen_bytes(len: uint) -> [u8] {
        let mut v = [];
        let mut i = 0u;
        while i < len {
            let n = self.next() as uint;
            v += [(n % (u8::max_value as uint)) as u8];
            i += 1u;
        }
        v
    }
}

#[doc = "Create a random number generator"]
fn rng() -> rng {
    resource rand_res(c: *rctx) { rustrt::rand_free(c); }

    impl of rng for @rand_res {
        fn next() -> u32 { ret rustrt::rand_next(**self); }
    }

    @rand_res(rustrt::rand_new()) as rng
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        let r1 = rand::rng();
        log(debug, r1.next());
        log(debug, r1.next());
        {
            let r2 = rand::rng();
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
    fn gen_float() {
        let r = rand::rng();
        let a = r.gen_float();
        let b = r.gen_float();
        log(debug, (a, b));
    }

    #[test]
    fn gen_str() {
        let r = rand::rng();
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        log(debug, r.gen_str(10u));
        assert r.gen_str(0u).len() == 0u;
        assert r.gen_str(10u).len() == 10u;
        assert r.gen_str(16u).len() == 16u;
    }

    #[test]
    fn gen_bytes() {
        let r = rand::rng();
        assert r.gen_bytes(0u).len() == 0u;
        assert r.gen_bytes(10u).len() == 10u;
        assert r.gen_bytes(16u).len() == 16u;
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

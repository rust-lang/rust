#[doc = "Random number generation"];

export rng, seed, seeded_rng, weighted, extensions;

enum rctx {}

#[abi = "cdecl"]
native mod rustrt {
    fn rand_seed() -> [u8];
    fn rand_new() -> *rctx;
    fn rand_new_seeded(seed: [u8]) -> *rctx;
    fn rand_next(c: *rctx) -> u32;
    fn rand_free(c: *rctx);
}

#[doc = "A random number generator"]
iface rng {
    #[doc = "Return the next random integer"]
    fn next() -> u32;
}

#[doc = "A value with a particular weight compared to other values"]
type weighted<T> = { weight: uint, item: T };

#[doc = "Extension methods for random number generators"]
impl extensions for rng {

    #[doc = "Return a random int"]
    fn gen_int() -> int {
        self.gen_i64() as int
    }

    #[doc = "Return an int randomly chosen from the range [start, end), \
             failing if start >= end"]
    fn gen_int_range(start: int, end: int) -> int {
        assert start < end;
        start + int::abs(self.gen_int() % (end - start))
    }

    #[doc = "Return a random i8"]
    fn gen_i8() -> i8 {
        self.next() as i8
    }

    #[doc = "Return a random i16"]
    fn gen_i16() -> i16 {
        self.next() as i16
    }

    #[doc = "Return a random i32"]
    fn gen_i32() -> i32 {
        self.next() as i32
    }

    #[doc = "Return a random i64"]
    fn gen_i64() -> i64 {
        (self.next() as i64 << 32) | self.next() as i64
    }

    #[doc = "Return a random uint"]
    fn gen_uint() -> uint {
        self.gen_u64() as uint
    }

    #[doc = "Return a uint randomly chosen from the range [start, end), \
             failing if start >= end"]
    fn gen_uint_range(start: uint, end: uint) -> uint {
        assert start < end;
        start + (self.gen_uint() % (end - start))
    }

    #[doc = "Return a random u8"]
    fn gen_u8() -> u8 {
        self.next() as u8
    }

    #[doc = "Return a random u16"]
    fn gen_u16() -> u16 {
        self.next() as u16
    }

    #[doc = "Return a random u32"]
    fn gen_u32() -> u32 {
        self.next()
    }

    #[doc = "Return a random u64"]
    fn gen_u64() -> u64 {
        (self.next() as u64 << 32) | self.next() as u64
    }

    #[doc = "Return a random float"]
    fn gen_float() -> float {
        self.gen_f64() as float
    }

    #[doc = "Return a random f32"]
    fn gen_f32() -> f32 {
        self.gen_f64() as f32
    }

    #[doc = "Return a random f64"]
    fn gen_f64() -> f64 {
        let u1 = self.next() as f64;
        let u2 = self.next() as f64;
        let u3 = self.next() as f64;
        let scale = u32::max_value as f64;
        ret ((u1 / scale + u2) / scale + u3) / scale;
    }

    #[doc = "Return a random char"]
    fn gen_char() -> char {
        self.next() as char
    }

    #[doc = "Return a char randomly chosen from chars, failing if chars is \
             empty"]
    fn gen_char_from(chars: str) -> char {
        assert !chars.is_empty();
        self.choose(str::chars(chars))
    }

    #[doc = "Return a random bool"]
    fn gen_bool() -> bool {
        self.next() & 1u32 == 1u32
    }

    #[doc = "Return a bool with a 1 in n chance of true"]
    fn gen_weighted_bool(n: uint) -> bool {
        if n == 0u {
            true
        } else {
            self.gen_uint_range(1u, n + 1u) == 1u
        }
    }

    #[doc = "Return a random string of the specified length composed of A-Z, \
             a-z, 0-9"]
    fn gen_str(len: uint) -> str {
        let charset = "ABCDEFGHIJKLMNOPQRSTUVWXYZ" +
                      "abcdefghijklmnopqrstuvwxyz" +
                      "0123456789";
        let mut s = "";
        let mut i = 0u;
        while (i < len) {
            s = s + str::from_char(self.gen_char_from(charset));
            i += 1u;
        }
        s
    }

    #[doc = "Return a random byte string of the specified length"]
    fn gen_bytes(len: uint) -> [u8] {
        vec::from_fn(len) {|_i|
            self.gen_u8()
        }
    }

    #[doc = "Choose an item randomly, failing if values is empty"]
    fn choose<T:copy>(values: [T]) -> T {
        self.choose_option(values).get()
    }

    #[doc = "Choose some(item) randomly, returning none if values is empty"]
    fn choose_option<T:copy>(values: [T]) -> option<T> {
        if values.is_empty() {
            none
        } else {
            some(values[self.gen_uint_range(0u, values.len())])
        }
    }

    #[doc = "Choose an item respecting the relative weights, failing if \
             the sum of the weights is 0"]
    fn choose_weighted<T: copy>(v : [weighted<T>]) -> T {
        self.choose_weighted_option(v).get()
    }

    #[doc = "Choose some(item) respecting the relative weights, returning \
             none if the sum of the weights is 0"]
    fn choose_weighted_option<T:copy>(v: [weighted<T>]) -> option<T> {
        let mut total = 0u;
        for v.each {|item|
            total += item.weight;
        }
        if total == 0u {
            ret none;
        }
        let chosen = self.gen_uint_range(0u, total);
        let mut so_far = 0u;
        for v.each {|item|
            so_far += item.weight;
            if so_far > chosen {
                ret some(item.item);
            }
        }
        unreachable();
    }

    #[doc = "Return a vec containing copies of the items, in order, where \
             the weight of the item determines how many copies there are"]
    fn weighted_vec<T:copy>(v: [weighted<T>]) -> [T] {
        let mut r = [];
        for v.each {|item|
            uint::range(0u, item.weight) {|_i|
                r += [item.item];
            }
        }
        r
    }

    #[doc = "Shuffle a vec"]
    fn shuffle<T:copy>(values: [T]) -> [T] {
        let mut m = vec::to_mut(values);
        self.shuffle_mut(m);
        ret vec::from_mut(m);
    }

    #[doc = "Shuffle a mutable vec in place"]
    fn shuffle_mut<T>(&values: [mut T]) {
        let mut i = values.len();
        while i >= 2u {
            // invariant: elements with index >= i have been locked in place.
            i -= 1u;
            // lock element i in place.
            vec::swap(values, i, self.gen_uint_range(0u, i + 1u));
        }
    }

}

resource rand_res(c: *rctx) { rustrt::rand_free(c); }

impl of rng for @rand_res {
    fn next() -> u32 { ret rustrt::rand_next(**self); }
}

#[doc = "Create a new random seed for seeded_rng"]
fn seed() -> [u8] {
    rustrt::rand_seed()
}

#[doc = "Create a random number generator with a system specified seed"]
fn rng() -> rng {
    @rand_res(rustrt::rand_new()) as rng
}

#[doc = "Create a random number generator using the specified seed. A \
         generator constructed with a given seed will generate the same \
         sequence of values as all other generators constructed with the \
         same seed. The seed may be any length."]
fn seeded_rng(seed: [u8]) -> rng {
    @rand_res(rustrt::rand_new_seeded(seed)) as rng
}

#[cfg(test)]
mod tests {

    #[test]
    fn rng_seeded() {
        let seed = rand::seed();
        let ra = rand::seeded_rng(seed);
        let rb = rand::seeded_rng(seed);
        assert ra.gen_str(100u) == rb.gen_str(100u);
    }

    #[test]
    fn rng_seeded_custom_seed() {
        // much shorter than generated seeds which are 1024 bytes
        let seed = [2u8, 32u8, 4u8, 32u8, 51u8];
        let ra = rand::seeded_rng(seed);
        let rb = rand::seeded_rng(seed);
        assert ra.gen_str(100u) == rb.gen_str(100u);
    }

    #[test]
    fn gen_int_range() {
        let r = rand::rng();
        let a = r.gen_int_range(-3, 42);
        assert a >= -3 && a < 42;
        assert r.gen_int_range(0, 1) == 0;
        assert r.gen_int_range(-12, -11) == -12;
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win3"))]
    fn gen_int_from_fail() {
        rand::rng().gen_int_range(5, -2);
    }

    #[test]
    fn gen_uint_range() {
        let r = rand::rng();
        let a = r.gen_uint_range(3u, 42u);
        assert a >= 3u && a < 42u;
        assert r.gen_uint_range(0u, 1u) == 0u;
        assert r.gen_uint_range(12u, 13u) == 12u;
    }

    #[test]
    #[should_fail]
    #[ignore(cfg(target_os = "win3"))]
    fn gen_uint_range_fail() {
        rand::rng().gen_uint_range(5u, 2u);
    }

    #[test]
    fn gen_float() {
        let r = rand::rng();
        let a = r.gen_float();
        let b = r.gen_float();
        log(debug, (a, b));
    }

    #[test]
    fn gen_weighted_bool() {
        let r = rand::rng();
        assert r.gen_weighted_bool(0u) == true;
        assert r.gen_weighted_bool(1u) == true;
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

    #[test]
    fn choose() {
        let r = rand::rng();
        assert r.choose([1, 1, 1]) == 1;
    }

    #[test]
    fn choose_option() {
        let r = rand::rng();
        assert r.choose_option([]) == none::<int>;
        assert r.choose_option([1, 1, 1]) == some(1);
    }

    #[test]
    fn choose_weighted() {
        let r = rand::rng();
        assert r.choose_weighted([{weight: 1u, item: 42}]) == 42;
        assert r.choose_weighted([
            {weight: 0u, item: 42},
            {weight: 1u, item: 43}
        ]) == 43;
    }

    #[test]
    fn choose_weighted_option() {
        let r = rand::rng();
        assert r.choose_weighted_option([{weight: 1u, item: 42}]) == some(42);
        assert r.choose_weighted_option([
            {weight: 0u, item: 42},
            {weight: 1u, item: 43}
        ]) == some(43);
        assert r.choose_weighted_option([]) == none::<int>;
    }

    #[test]
    fn weighted_vec() {
        let r = rand::rng();
        let empty: [int] = [];
        assert r.weighted_vec([]) == empty;
        assert r.weighted_vec([
            {weight: 0u, item: 3u},
            {weight: 1u, item: 2u},
            {weight: 2u, item: 1u}
        ]) == [2u, 1u, 1u];
    }

    #[test]
    fn shuffle() {
        let r = rand::rng();
        let empty: [int] = [];
        assert r.shuffle([]) == empty;
        assert r.shuffle([1, 1, 1]) == [1, 1, 1];
    }
}


// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

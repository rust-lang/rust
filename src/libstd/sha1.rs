/*
Module: sha1

An implementation of the SHA-1 cryptographic hash.

First create a <sha1> object using the <mk_sha1> constructor, then
feed it input using the <input> or <input_str> methods, which may be
called any number of times.

After the entire input has been fed to the hash read the result using
the <result> or <result_str> methods.

The <sha1> object may be reused to create multiple hashes by calling
the <reset> method.
*/

/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */
export sha1;
export mk_sha1;

/* Section: Types */

/*
Iface: sha1

The SHA-1 interface
*/
iface sha1 {
    /*
    Method: input

    Provide message input as bytes
    */
    fn input([u8]);
    /*
    Method: input_str

    Provide message input as string
    */
    fn input_str(str);
    /*
    Method: result

    Read the digest as a vector of 20 bytes. After calling this no further
    input may be provided until reset is called.
    */
    fn result() -> [u8];
    /*
    Method: result_str

    Read the digest as a hex string. After calling this no further
    input may be provided until reset is called.
    */
    fn result_str() -> str;
    /*
    Method: reset

    Reset the SHA-1 state for reuse
    */
    fn reset();
}

/* Section: Operations */

// Some unexported constants
const digest_buf_len: uint = 5u;
const msg_block_len: uint = 64u;
const work_buf_len: uint = 80u;
const k0: u32 = 0x5A827999u32;
const k1: u32 = 0x6ED9EBA1u32;
const k2: u32 = 0x8F1BBCDCu32;
const k3: u32 = 0xCA62C1D6u32;


/*
Function: mk_sha1

Construct a <sha1> object
*/
fn mk_sha1() -> sha1 {
    type sha1state =
        {h: [mutable u32],
         mutable len_low: u32,
         mutable len_high: u32,
         msg_block: [mutable u8],
         mutable msg_block_idx: uint,
         mutable computed: bool,
         work_buf: [mutable u32]};

    fn add_input(st: sha1state, msg: [u8]) {
        // FIXME: Should be typestate precondition
        assert (!st.computed);
        for element: u8 in msg {
            st.msg_block[st.msg_block_idx] = element;
            st.msg_block_idx += 1u;
            st.len_low += 8u32;
            if st.len_low == 0u32 {
                st.len_high += 1u32;
                if st.len_high == 0u32 {
                    // FIXME: Need better failure mode

                    fail;
                }
            }
            if st.msg_block_idx == msg_block_len { process_msg_block(st); }
        }
    }
    fn process_msg_block(st: sha1state) {
        // FIXME: Make precondition
        assert (vec::len(st.h) == digest_buf_len);
        assert (vec::len(st.work_buf) == work_buf_len);
        let t: int; // Loop counter
        let w = st.work_buf;

        // Initialize the first 16 words of the vector w
        t = 0;
        while t < 16 {
            let tmp;
            tmp = (st.msg_block[t * 4] as u32) << 24u32;
            tmp = tmp | (st.msg_block[t * 4 + 1] as u32) << 16u32;
            tmp = tmp | (st.msg_block[t * 4 + 2] as u32) << 8u32;
            tmp = tmp | (st.msg_block[t * 4 + 3] as u32);
            w[t] = tmp;
            t += 1;
        }

        // Initialize the rest of vector w
        while t < 80 {
            let val = w[t - 3] ^ w[t - 8] ^ w[t - 14] ^ w[t - 16];
            w[t] = circular_shift(1u32, val);
            t += 1;
        }
        let a = st.h[0];
        let b = st.h[1];
        let c = st.h[2];
        let d = st.h[3];
        let e = st.h[4];
        let temp: u32;
        t = 0;
        while t < 20 {
            temp = circular_shift(5u32, a) + (b & c | !b & d) + e + w[t] + k0;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }
        while t < 40 {
            temp = circular_shift(5u32, a) + (b ^ c ^ d) + e + w[t] + k1;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }
        while t < 60 {
            temp =
                circular_shift(5u32, a) + (b & c | b & d | c & d) + e + w[t] +
                    k2;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }
        while t < 80 {
            temp = circular_shift(5u32, a) + (b ^ c ^ d) + e + w[t] + k3;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }
        st.h[0] = st.h[0] + a;
        st.h[1] = st.h[1] + b;
        st.h[2] = st.h[2] + c;
        st.h[3] = st.h[3] + d;
        st.h[4] = st.h[4] + e;
        st.msg_block_idx = 0u;
    }
    fn circular_shift(bits: u32, word: u32) -> u32 {
        ret word << bits | word >> 32u32 - bits;
    }
    fn mk_result(st: sha1state) -> [u8] {
        if !st.computed { pad_msg(st); st.computed = true; }
        let rs: [u8] = [];
        for hpart: u32 in st.h {
            let a = hpart >> 24u32 & 0xFFu32 as u8;
            let b = hpart >> 16u32 & 0xFFu32 as u8;
            let c = hpart >> 8u32 & 0xFFu32 as u8;
            let d = hpart & 0xFFu32 as u8;
            rs += [a, b, c, d];
        }
        ret rs;
    }

    /*
     * According to the standard, the message must be padded to an even
     * 512 bits.  The first padding bit must be a '1'.  The last 64 bits
     * represent the length of the original message.  All bits in between
     * should be 0.  This function will pad the message according to those
     * rules by filling the msg_block vector accordingly.  It will also
     * call process_msg_block() appropriately.  When it returns, it
     * can be assumed that the message digest has been computed.
     */
    fn pad_msg(st: sha1state) {
        // FIXME: Should be a precondition
        assert (vec::len(st.msg_block) == msg_block_len);

        /*
         * Check to see if the current message block is too small to hold
         * the initial padding bits and length.  If so, we will pad the
         * block, process it, and then continue padding into a second block.
         */
        if st.msg_block_idx > 55u {
            st.msg_block[st.msg_block_idx] = 0x80u8;
            st.msg_block_idx += 1u;
            while st.msg_block_idx < msg_block_len {
                st.msg_block[st.msg_block_idx] = 0u8;
                st.msg_block_idx += 1u;
            }
            process_msg_block(st);
        } else {
            st.msg_block[st.msg_block_idx] = 0x80u8;
            st.msg_block_idx += 1u;
        }
        while st.msg_block_idx < 56u {
            st.msg_block[st.msg_block_idx] = 0u8;
            st.msg_block_idx += 1u;
        }

        // Store the message length as the last 8 octets
        st.msg_block[56] = st.len_high >> 24u32 & 0xFFu32 as u8;
        st.msg_block[57] = st.len_high >> 16u32 & 0xFFu32 as u8;
        st.msg_block[58] = st.len_high >> 8u32 & 0xFFu32 as u8;
        st.msg_block[59] = st.len_high & 0xFFu32 as u8;
        st.msg_block[60] = st.len_low >> 24u32 & 0xFFu32 as u8;
        st.msg_block[61] = st.len_low >> 16u32 & 0xFFu32 as u8;
        st.msg_block[62] = st.len_low >> 8u32 & 0xFFu32 as u8;
        st.msg_block[63] = st.len_low & 0xFFu32 as u8;
        process_msg_block(st);
    }

    impl of sha1 for sha1state {
        fn reset() {
            // FIXME: Should be typestate precondition
            assert (vec::len(self.h) == digest_buf_len);
            self.len_low = 0u32;
            self.len_high = 0u32;
            self.msg_block_idx = 0u;
            self.h[0] = 0x67452301u32;
            self.h[1] = 0xEFCDAB89u32;
            self.h[2] = 0x98BADCFEu32;
            self.h[3] = 0x10325476u32;
            self.h[4] = 0xC3D2E1F0u32;
            self.computed = false;
        }
        fn input(msg: [u8]) { add_input(self, msg); }
        fn input_str(msg: str) { add_input(self, str::bytes(msg)); }
        fn result() -> [u8] { ret mk_result(self); }
        fn result_str() -> str {
            let r = mk_result(self);
            let s = "";
            for b: u8 in r { s += uint::to_str(b as uint, 16u); }
            ret s;
        }
    }
    let st = {
        h: vec::init_elt_mut(digest_buf_len, 0u32),
        mutable len_low: 0u32,
        mutable len_high: 0u32,
        msg_block: vec::init_elt_mut(msg_block_len, 0u8),
        mutable msg_block_idx: 0u,
        mutable computed: false,
        work_buf: vec::init_elt_mut(work_buf_len, 0u32)
    };
    let sh = st as sha1;
    sh.reset();
    ret sh;
}

#[cfg(test)]
mod tests {

    #[test]
    fn test() {
        type test = {input: str, output: [u8]};

        fn a_million_letter_a() -> str {
            let i = 0;
            let rs = "";
            while i < 100000 { rs += "aaaaaaaaaa"; i += 1; }
            ret rs;
        }
        // Test messages from FIPS 180-1

        let fips_180_1_tests: [test] =
            [{input: "abc",
              output:
                  [0xA9u8, 0x99u8, 0x3Eu8, 0x36u8,
                   0x47u8, 0x06u8, 0x81u8, 0x6Au8,
                   0xBAu8, 0x3Eu8, 0x25u8, 0x71u8,
                   0x78u8, 0x50u8, 0xC2u8, 0x6Cu8,
                   0x9Cu8, 0xD0u8, 0xD8u8, 0x9Du8]},
             {input:
                  "abcdbcdecdefdefgefghfghighij" +
                  "hijkijkljklmklmnlmnomnopnopq",
              output:
                  [0x84u8, 0x98u8, 0x3Eu8, 0x44u8,
                   0x1Cu8, 0x3Bu8, 0xD2u8, 0x6Eu8,
                   0xBAu8, 0xAEu8, 0x4Au8, 0xA1u8,
                   0xF9u8, 0x51u8, 0x29u8, 0xE5u8,
                   0xE5u8, 0x46u8, 0x70u8, 0xF1u8]},
             {input: a_million_letter_a(),
              output:
                  [0x34u8, 0xAAu8, 0x97u8, 0x3Cu8,
                   0xD4u8, 0xC4u8, 0xDAu8, 0xA4u8,
                   0xF6u8, 0x1Eu8, 0xEBu8, 0x2Bu8,
                   0xDBu8, 0xADu8, 0x27u8, 0x31u8,
                   0x65u8, 0x34u8, 0x01u8, 0x6Fu8]}];
        // Examples from wikipedia

        let wikipedia_tests: [test] =
            [{input: "The quick brown fox jumps over the lazy dog",
              output:
                  [0x2fu8, 0xd4u8, 0xe1u8, 0xc6u8,
                   0x7au8, 0x2du8, 0x28u8, 0xfcu8,
                   0xedu8, 0x84u8, 0x9eu8, 0xe1u8,
                   0xbbu8, 0x76u8, 0xe7u8, 0x39u8,
                   0x1bu8, 0x93u8, 0xebu8, 0x12u8]},
             {input: "The quick brown fox jumps over the lazy cog",
              output:
                  [0xdeu8, 0x9fu8, 0x2cu8, 0x7fu8,
                   0xd2u8, 0x5eu8, 0x1bu8, 0x3au8,
                   0xfau8, 0xd3u8, 0xe8u8, 0x5au8,
                   0x0bu8, 0xd1u8, 0x7du8, 0x9bu8,
                   0x10u8, 0x0du8, 0xb4u8, 0xb3u8]}];
        let tests = fips_180_1_tests + wikipedia_tests;
        fn check_vec_eq(v0: [u8], v1: [u8]) {
            assert (vec::len::<u8>(v0) == vec::len::<u8>(v1));
            let len = vec::len::<u8>(v0);
            let i = 0u;
            while i < len {
                let a = v0[i];
                let b = v1[i];
                assert (a == b);
                i += 1u;
            }
        }
        // Test that it works when accepting the message all at once

        let sh = sha1::mk_sha1();
        for t: test in tests {
            sh.input_str(t.input);
            let out = sh.result();
            check_vec_eq(t.output, out);
            sh.reset();
        }


        // Test that it works when accepting the message in pieces
        for t: test in tests {
            let len = str::byte_len(t.input);
            let left = len;
            while left > 0u {
                let take = (left + 1u) / 2u;
                sh.input_str(str::substr(t.input, len - left, take));
                left = left - take;
            }
            let out = sh.result();
            check_vec_eq(t.output, out);
            sh.reset();
        }
    }

}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// End:

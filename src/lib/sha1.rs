/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */
export sha1;
export mk_sha1;

type sha1 = obj {
    // Provide message input as bytes
    fn input(&[u8]);
    // Provide message input as string
    fn input_str(&istr);
    // Read the digest as a vector of 20 bytes. After calling this no further
    // input may provided until reset is called
    fn result() -> [u8];
    // Same as above, just a hex-string version.
    fn result_str() -> istr;
    // Reset the sha1 state for reuse. This is called
    // automatically during construction
    fn reset();
};


// Some unexported constants
const digest_buf_len: uint = 5u;
const msg_block_len: uint = 64u;
const work_buf_len: uint = 80u;
const k0: u32 = 0x5A827999u32;
const k1: u32 = 0x6ED9EBA1u32;
const k2: u32 = 0x8F1BBCDCu32;
const k3: u32 = 0xCA62C1D6u32;


// Builds a sha1 object
fn mk_sha1() -> sha1 {
    type sha1state =
        {h: [mutable u32],
         mutable len_low: u32,
         mutable len_high: u32,
         msg_block: [mutable u8],
         mutable msg_block_idx: uint,
         mutable computed: bool,
         work_buf: [mutable u32]};

    fn add_input(st: &sha1state, msg: &[u8]) {
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
    fn process_msg_block(st: &sha1state) {
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
    fn mk_result(st: &sha1state) -> [u8] {
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
    fn pad_msg(st: &sha1state) {
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
    obj sha1(st: sha1state) {
        fn reset() {
            // FIXME: Should be typestate precondition
            assert (vec::len(st.h) == digest_buf_len);
            st.len_low = 0u32;
            st.len_high = 0u32;
            st.msg_block_idx = 0u;
            st.h[0] = 0x67452301u32;
            st.h[1] = 0xEFCDAB89u32;
            st.h[2] = 0x98BADCFEu32;
            st.h[3] = 0x10325476u32;
            st.h[4] = 0xC3D2E1F0u32;
            st.computed = false;
        }
        fn input(msg: &[u8]) { add_input(st, msg); }
        fn input_str(msg: &istr) { add_input(st, istr::bytes(msg)); }
        fn result() -> [u8] { ret mk_result(st); }
        fn result_str() -> istr {
            let r = mk_result(st);
            let s = ~"";
            for b: u8 in r {
                s += uint::to_str(b as uint, 16u);
            }
            ret s;
        }
    }
    let st =
        {h: vec::init_elt_mut::<u32>(0u32, digest_buf_len),
         mutable len_low: 0u32,
         mutable len_high: 0u32,
         msg_block: vec::init_elt_mut::<u8>(0u8, msg_block_len),
         mutable msg_block_idx: 0u,
         mutable computed: false,
         work_buf: vec::init_elt_mut::<u32>(0u32, work_buf_len)};
    let sh = sha1(st);
    sh.reset();
    ret sh;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C $RBUILD 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

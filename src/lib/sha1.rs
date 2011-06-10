/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */

export sha1;
export mk_sha1;

state type sha1 = state obj {
                        // Provide message input as bytes
                        fn input(&vec[u8]);

                        // Provide message input as string
                        fn input_str(&str);

                        // Read the digest as a vector of 20 bytes. After
                        // calling this no further input may provided
                        // until reset is called
                        fn result() -> vec[u8];

                        // Same as above, just a hex-string version.
                        fn result_str() -> str;

                        // Reset the sha1 state for reuse. This is called
                        // automatically during construction
                        fn reset();
};

// Some unexported constants
const uint digest_buf_len = 5u;
const uint msg_block_len = 64u;
const uint work_buf_len = 80u;

const u32 k0 = 0x5A827999u32;
const u32 k1 = 0x6ED9EBA1u32;
const u32 k2 = 0x8F1BBCDCu32;
const u32 k3 = 0xCA62C1D6u32;

// Builds a sha1 object
fn mk_sha1() -> sha1 {

    state type sha1state = rec(vec[mutable u32] h,
                               mutable u32 len_low,
                               mutable u32 len_high,
                               vec[mutable u8] msg_block,
                               mutable uint msg_block_idx,
                               mutable bool computed,
                               vec[mutable u32] work_buf);

    fn add_input(&sha1state st, &vec[u8] msg) {
        // FIXME: Should be typestate precondition
        assert (!st.computed);

        for (u8 element in msg) {
            st.msg_block.(st.msg_block_idx) = element;
            st.msg_block_idx += 1u;

            st.len_low += 8u32;
            if (st.len_low == 0u32) {
                st.len_high += 1u32;
                if (st.len_high == 0u32) {
                    // FIXME: Need better failure mode
                    fail;
                }
            }

            if (st.msg_block_idx == msg_block_len) {
                process_msg_block(st);
            }
        }
    }

    fn process_msg_block(&sha1state st) {

        // FIXME: Make precondition
        assert (vec::len(st.h) == digest_buf_len);
        assert (vec::len(st.work_buf) == work_buf_len);

        let int t; // Loop counter
        auto w = st.work_buf;

        // Initialize the first 16 words of the vector w
        t = 0;
        while (t < 16) {
            auto tmp;
            tmp = (st.msg_block.(t * 4) as u32) << 24u32;
            tmp = tmp | ((st.msg_block.(t * 4 + 1) as u32) << 16u32);
            tmp = tmp | ((st.msg_block.(t * 4 + 2) as u32) << 8u32);
            tmp = tmp | (st.msg_block.(t * 4 + 3) as u32);
            w.(t) = tmp;
            t += 1;
        }

        // Initialize the rest of vector w
        while (t < 80) {
            auto val = w.(t-3) ^ w.(t-8) ^ w.(t-14) ^ w.(t-16);
            w.(t) = circular_shift(1u32, val);
            t += 1;
        }

        auto a = st.h.(0);
        auto b = st.h.(1);
        auto c = st.h.(2);
        auto d = st.h.(3);
        auto e = st.h.(4);

        let u32 temp;

        t = 0;
        while (t < 20) {
            temp = circular_shift(5u32, a)
                + ((b & c) | ((!b) & d)) + e + w.(t) + k0;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 40) {
            temp = circular_shift(5u32, a)
                + (b ^ c ^ d) + e + w.(t) + k1;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 60) {
            temp = circular_shift(5u32, a)
                + ((b & c) | (b & d) | (c & d)) + e + w.(t) + k2;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 80) {
            temp = circular_shift(5u32, a)
                + (b ^ c ^ d) + e + w.(t) + k3;
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        st.h.(0) = st.h.(0) + a;
        st.h.(1) = st.h.(1) + b;
        st.h.(2) = st.h.(2) + c;
        st.h.(3) = st.h.(3) + d;
        st.h.(4) = st.h.(4) + e;

        st.msg_block_idx = 0u;
    }

    fn circular_shift(u32 bits, u32 word) -> u32 {
        // FIXME: This is a workaround for a rustboot
        // "unrecognized quads" codegen bug
        auto bits_hack = bits;
        ret (word << bits_hack) | (word >> (32u32 - bits));
    }

    fn mk_result(&sha1state st) -> vec[u8] {
        if (!st.computed) {
            pad_msg(st);
            st.computed = true;
        }

        let vec[u8] res = [];
        for (u32 hpart in st.h) {
            auto a = (hpart >> 24u32) & 0xFFu32 as u8;
            auto b = (hpart >> 16u32) & 0xFFu32 as u8;
            auto c = (hpart >> 8u32) & 0xFFu32 as u8;
            auto d = (hpart & 0xFFu32 as u8);
            res += [a,b,c,d];
        }
        ret res;
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
    fn pad_msg(&sha1state st) {
        // FIXME: Should be a precondition
        assert (vec::len(st.msg_block) == msg_block_len);

        /*
         * Check to see if the current message block is too small to hold
         * the initial padding bits and length.  If so, we will pad the
         * block, process it, and then continue padding into a second block.
         */
        if (st.msg_block_idx > 55u) {
            st.msg_block.(st.msg_block_idx) = 0x80u8;
            st.msg_block_idx += 1u;

            while (st.msg_block_idx < msg_block_len) {
                st.msg_block.(st.msg_block_idx) = 0u8;
                st.msg_block_idx += 1u;
            }

            process_msg_block(st);
        } else {
            st.msg_block.(st.msg_block_idx) = 0x80u8;
            st.msg_block_idx += 1u;
        }

        while (st.msg_block_idx < 56u) {
            st.msg_block.(st.msg_block_idx) = 0u8;
            st.msg_block_idx += 1u;
        }

        // Store the message length as the last 8 octets
        st.msg_block.(56) = (st.len_high >> 24u32) & 0xFFu32 as u8;
        st.msg_block.(57) = (st.len_high >> 16u32) & 0xFFu32 as u8;
        st.msg_block.(58) = (st.len_high >> 8u32) & 0xFFu32 as u8;
        st.msg_block.(59) = st.len_high & 0xFFu32 as u8;
        st.msg_block.(60) = (st.len_low >> 24u32) & 0xFFu32 as u8;
        st.msg_block.(61) = (st.len_low >> 16u32) & 0xFFu32 as u8;
        st.msg_block.(62) = (st.len_low >> 8u32) & 0xFFu32 as u8;
        st.msg_block.(63) = st.len_low & 0xFFu32 as u8;

        process_msg_block(st);
    }

    state obj sha1(sha1state st) {

        fn reset() {
            // FIXME: Should be typestate precondition
            assert (vec::len(st.h) == digest_buf_len);

            st.len_low = 0u32;
            st.len_high = 0u32;
            st.msg_block_idx = 0u;

            st.h.(0) = 0x67452301u32;
            st.h.(1) = 0xEFCDAB89u32;
            st.h.(2) = 0x98BADCFEu32;
            st.h.(3) = 0x10325476u32;
            st.h.(4) = 0xC3D2E1F0u32;

            st.computed = false;
        }

        fn input(&vec[u8] msg) {
            add_input(st, msg);
        }

        fn input_str(&str msg) {
            add_input(st, str::bytes(msg));
        }

        fn result() -> vec[u8] {
            ret mk_result(st);
        }

        fn result_str() -> str {
            auto r = mk_result(st);
            auto s = "";
            for (u8 b in r) {
                s += uint::to_str(b as uint, 16u);
            }
            ret s;
        }
    }

    auto st = rec(h = vec::init_elt_mut[u32](0u32, digest_buf_len),
                  mutable len_low = 0u32,
                  mutable len_high = 0u32,
                  msg_block = vec::init_elt_mut[u8](0u8, msg_block_len),
                  mutable msg_block_idx = 0u,
                  mutable computed = false,
                  work_buf = vec::init_elt_mut[u32](0u32, work_buf_len));
    auto sh = sha1(st);
    sh.reset();
    ret sh;
}

// Local Variables:
// mode: rust;
// fill-column: 78;
// indent-tabs-mode: nil
// c-basic-offset: 4
// buffer-file-coding-system: utf-8-unix
// compile-command: "make -k -C .. 2>&1 | sed -e 's/\\/x\\//x:\\//g'";
// End:

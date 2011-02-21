/*
 * A SHA-1 implementation derived from Paul E. Jones's reference
 * implementation, which is written for clarity, not speed. At some
 * point this will want to be rewritten.
 */

import std._vec;
import std._str;

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

                        // Reset the sha1 state for reuse. This is called
                        // automatically during construction
                        fn reset();
};

// Some unexported constants
const uint digest_buf_len = 5;
const uint msg_block_len = 64;

// Builds a sha1 object
fn mk_sha1() -> sha1 {

    state type sha1state = rec(vec[mutable u32] h,
                               mutable u32 len_low,
                               mutable u32 len_high,
                               vec[mutable u8] msg_block,
                               mutable uint msg_block_idx,
                               mutable bool computed);

    impure fn add_input(&sha1state st, &vec[u8] msg) {
        // FIXME: Should be typestate precondition
        check (!st.computed);

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

    impure fn process_msg_block(&sha1state st) {

        // FIXME: Make precondition
        check (_vec.len[mutable u32](st.h) == digest_buf_len);

        // Constants
        auto k = vec(0x5A827999u32,
                     0x6ED9EBA1u32,
                     0x8F1BBCDCu32,
                     0xCA62C1D6u32);

        let int t; // Loop counter
        let vec[mutable u32] w = _vec.init_elt[mutable u32](0u32, 80u);

        // Initialize the first 16 words of the vector w
        t = 0;
        while (t < 16) {
            w.(t) = (st.msg_block.(t * 4) as u32) << 24u32;
            w.(t) = w.(t) | ((st.msg_block.(t * 4 + 1) as u32) << 16u32);
            w.(t) = w.(t) | ((st.msg_block.(t * 4 + 2) as u32) << 8u32);
            w.(t) = w.(t) | (st.msg_block.(t * 4 + 3) as u32);
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
                + ((b & c) | ((~b) & d)) + e + w.(t) + k.(0);
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 40) {
            temp = circular_shift(5u32, a)
                + (b ^ c ^ d) + e + w.(t) + k.(1);
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 60) {
            temp = circular_shift(5u32, a)
                + ((b & c) | (b & d) | (c & d)) + e + w.(t) + k.(2);
            e = d;
            d = c;
            c = circular_shift(30u32, b);
            b = a;
            a = temp;
            t += 1;
        }

        while (t < 80) {
            temp = circular_shift(5u32, a)
                + (b ^ c ^ d) + e + w.(t) + k.(3);
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

    impure fn mk_result(&sha1state st) -> vec[u8] {
        if (!st.computed) {
            pad_msg(st);
            st.computed = true;
        }

        let vec[u8] res = vec();
        for (u32 hpart in st.h) {
            res += (hpart >> 24u32) & 0xFFu32 as u8;
            res += (hpart >> 16u32) & 0xFFu32 as u8;
            res += (hpart >> 8u32) & 0xFFu32 as u8;
            res += hpart & 0xFFu32 as u8;
        }
        ret res;
    }

    /*
     * According to the standard, the message must be padded to an even
     * 512 bits.  The first padding bit must be a '1'.  The last 64 bits
     * represent the length of the original message.  All bits in between
     * should be 0.  This function will pad the message according to those
     * rules by filling the message_block array accordingly.  It will also
     * call ProcessMessageBlock() appropriately.  When it returns, it
     * can be assumed that the message digest has been computed.
     */
    impure fn pad_msg(&sha1state st) {
        // FIXME: Should be a precondition
        check (_vec.len[mutable u8](st.msg_block) == msg_block_len);

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
            check (_vec.len[mutable u32](st.h) == digest_buf_len);

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
            add_input(st, _str.bytes(msg));
        }

        fn result() -> vec[u8] {
            ret mk_result(st);
        }
    }

    auto st = rec(h = _vec.init_elt[mutable u32](0u32, digest_buf_len),
                  mutable len_low = 0u32,
                  mutable len_high = 0u32,
                  msg_block = _vec.init_elt[mutable u8](0u8, msg_block_len),
                  mutable msg_block_idx = 0u,
                  mutable computed = false);
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

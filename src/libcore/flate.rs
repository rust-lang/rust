/*!

Simple compression

*/

// NB: transitionary, de-mode-ing.
#[forbid(deprecated_mode)];
#[forbid(deprecated_pattern)];

use libc::{c_void, size_t, c_int};

extern mod rustrt {
    #[legacy_exports];

    fn tdefl_compress_mem_to_heap(psrc_buf: *const c_void,
                                  src_buf_len: size_t,
                                  pout_len: *size_t,
                                  flags: c_int) -> *c_void;

    fn tinfl_decompress_mem_to_heap(psrc_buf: *const c_void,
                                    src_buf_len: size_t,
                                    pout_len: *size_t,
                                    flags: c_int) -> *c_void;
}

const lz_none : c_int = 0x0;   // Huffman-coding only.
const lz_fast : c_int = 0x1;   // LZ with only one probe
const lz_norm : c_int = 0x80;  // LZ with 128 probes, "normal"
const lz_best : c_int = 0xfff; // LZ with 4095 probes, "best"

fn deflate_bytes(bytes: &[const u8]) -> ~[u8] {
    do vec::as_const_buf(bytes) |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tdefl_compress_mem_to_heap(b as *c_void,
                                                   len as size_t,
                                                   ptr::addr_of(outsz),
                                                   lz_norm);
            assert res as int != 0;
            let out = vec::raw::from_buf(res as *u8,
                                            outsz as uint);
            libc::free(res);
            move out
        }
    }
}

fn inflate_bytes(bytes: &[const u8]) -> ~[u8] {
    do vec::as_const_buf(bytes) |b, len| {
        unsafe {
            let mut outsz : size_t = 0;
            let res =
                rustrt::tinfl_decompress_mem_to_heap(b as *c_void,
                                                     len as size_t,
                                                     ptr::addr_of(outsz),
                                                     0);
            assert res as int != 0;
            let out = vec::raw::from_buf(res as *u8,
                                            outsz as uint);
            libc::free(res);
            move out
        }
    }
}

#[test]
#[allow(non_implicitly_copyable_typarams)]
fn test_flate_round_trip() {
    let r = rand::Rng();
    let mut words = ~[];
    for 20.times {
        words.push(r.gen_bytes(r.gen_uint_range(1, 10)));
    }
    for 20.times {
        let mut in = ~[];
        for 2000.times {
            in.push_all(r.choose(words));
        }
        debug!("de/inflate of %u bytes of random word-sequences",
               in.len());
        let cmp = flate::deflate_bytes(in);
        let out = flate::inflate_bytes(cmp);
        debug!("%u bytes deflated to %u (%.1f%% size)",
               in.len(), cmp.len(),
               100.0 * ((cmp.len() as float) / (in.len() as float)));
        assert(in == out);
    }
}

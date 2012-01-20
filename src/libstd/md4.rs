fn md4(msg: [u8]) -> {a: u32, b: u32, c: u32, d: u32} {
    // subtle: if orig_len is merely uint, then the code below
    // which performs shifts by 32 bits or more has undefined
    // results.
    let orig_len: u64 = (vec::len(msg) * 8u) as u64;

    // pad message
    let msg = msg + [0x80u8];
    let bitlen = orig_len + 8u64;
    while (bitlen + 64u64) % 512u64 > 0u64 {
        msg += [0u8];
        bitlen += 8u64;
    }

    // append length
    let i = 0u64;
    while i < 8u64 {
        msg += [(orig_len >> (i * 8u64)) as u8];
        i += 1u64;
    }

    let a = 0x67452301u32;
    let b = 0xefcdab89u32;
    let c = 0x98badcfeu32;
    let d = 0x10325476u32;

    fn rot(r: int, x: u32) -> u32 {
        let r = r as u32;
        (x << r) | (x >> (32u32 - r))
    }

    let i = 0u, e = vec::len(msg);
    let x = vec::init_elt_mut(16u, 0u32);
    while i < e {
        let aa = a, bb = b, cc = c, dd = d;

        let j = 0u, base = i;
        while j < 16u {
            x[j] = (msg[base] as u32) + (msg[base + 1u] as u32 << 8u32) +
                (msg[base + 2u] as u32 << 16u32) +
                (msg[base + 3u] as u32 << 24u32);
            j += 1u; base += 4u;
        }

        let j = 0u;
        while j < 16u {
            a = rot(3, a + ((b & c) | (!b & d)) + x[j]);
            j += 1u;
            d = rot(7, d + ((a & b) | (!a & c)) + x[j]);
            j += 1u;
            c = rot(11, c + ((d & a) | (!d & b)) + x[j]);
            j += 1u;
            b = rot(19, b + ((c & d) | (!c & a)) + x[j]);
            j += 1u;
        }

        let j = 0u, q = 0x5a827999u32;
        while j < 4u {
            a = rot(3, a + ((b & c) | ((b & d) | (c & d))) + x[j] + q);
            d = rot(5, d + ((a & b) | ((a & c) | (b & c))) + x[j + 4u] + q);
            c = rot(9, c + ((d & a) | ((d & b) | (a & b))) + x[j + 8u] + q);
            b = rot(13, b + ((c & d) | ((c & a) | (d & a))) + x[j + 12u] + q);
            j += 1u;
        }

        let j = 0u, q = 0x6ed9eba1u32;
        while j < 8u {
            let jj = j > 2u ? j - 3u : j;
            a = rot(3, a + (b ^ c ^ d) + x[jj] + q);
            d = rot(9, d + (a ^ b ^ c) + x[jj + 8u] + q);
            c = rot(11, c + (d ^ a ^ b) + x[jj + 4u] + q);
            b = rot(15, b + (c ^ d ^ a) + x[jj + 12u] + q);
            j += 2u;
        }

        a += aa; b += bb; c += cc; d += dd;
        i += 64u;
    }
    ret {a: a, b: b, c: c, d: d};
}

fn md4_str(msg: [u8]) -> str {
    let {a, b, c, d} = md4(msg);
    fn app(a: u32, b: u32, c: u32, d: u32, f: block(u32)) {
        f(a); f(b); f(c); f(d);
    }
    let result = "";
    app(a, b, c, d) {|u|
        let i = 0u32;
        while i < 4u32 {
            let byte = (u >> (i * 8u32)) as u8;
            if byte <= 16u8 { result += "0"; }
            result += uint::to_str(byte as uint, 16u);
            i += 1u32;
        }
    }
    result
}

fn md4_text(msg: str) -> str { md4_str(str::bytes(msg)) }

#[test]
fn test_md4() {
    assert md4_text("") == "31d6cfe0d16ae931b73c59d7e0c089c0";
    assert md4_text("a") == "bde52cb31de33e46245e05fbdbd6fb24";
    assert md4_text("abc") == "a448017aaf21d8525fc10ae87aa6729d";
    assert md4_text("message digest") == "d9130a8164549fe818874806e1c7014b";
    assert md4_text("abcdefghijklmnopqrstuvwxyz") ==
        "d79e1c308aa5bbcdeea8ed63df412da9";
    assert md4_text("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123\
                     456789") == "043f8582f241db351ce627e153e7f0e4";
    assert md4_text("12345678901234567890123456789012345678901234567890123456\
                     789012345678901234567890") ==
        "e33b4ddc9c38f2199c3e7b164fcc0536";
}

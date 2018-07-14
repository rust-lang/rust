// origin: FreeBSD /usr/src/lib/msun/src/s_exp2.c */
//-
// Copyright (c) 2005 David Schultz <das@FreeBSD.ORG>
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE.

use super::scalbn::scalbn;

const TBLSIZE: usize = 256;

// exp2(x): compute the base 2 exponential of x
//
// Accuracy: Peak error < 0.503 ulp for normalized results.
//
// Method: (accurate tables)
//
//   Reduce x:
//     x = k + y, for integer k and |y| <= 1/2.
//     Thus we have exp2(x) = 2**k * exp2(y).
//
//   Reduce y:
//     y = i/TBLSIZE + z - eps[i] for integer i near y * TBLSIZE.
//     Thus we have exp2(y) = exp2(i/TBLSIZE) * exp2(z - eps[i]),
//     with |z - eps[i]| <= 2**-9 + 2**-39 for the table used.
//
//   We compute exp2(i/TBLSIZE) via table lookup and exp2(z - eps[i]) via
//   a degree-5 minimax polynomial with maximum error under 1.3 * 2**-61.
//   The values in exp2t[] and eps[] are chosen such that
//   exp2t[i] = exp2(i/TBLSIZE + eps[i]), and eps[i] is a small offset such
//   that exp2t[i] is accurate to 2**-64.
//
//   Note that the range of i is +-TBLSIZE/2, so we actually index the tables
//   by i0 = i + TBLSIZE/2.  For cache efficiency, exp2t[] and eps[] are
//   virtual tables, interleaved in the real table tbl[].
//
//   This method is due to Gal, with many details due to Gal and Bachelis:
//
//      Gal, S. and Bachelis, B.  An Accurate Elementary Mathematical Library
//      for the IEEE Floating Point Standard.  TOMS 17(1), 26-46 (1991).
pub fn exp2(mut x: f64) -> f64 {
    let redux = f64::from_bits(0x4338000000000000) / TBLSIZE as f64;
    let p1 = f64::from_bits(0x3fe62e42fefa39ef);
    let p2 = f64::from_bits(0x3fcebfbdff82c575);
    let p3 = f64::from_bits(0x3fac6b08d704a0a6);
    let p4 = f64::from_bits(0x3f83b2ab88f70400);
    let p5 = f64::from_bits(0x3f55d88003875c74);

    #[cfg_attr(rustfmt, rustfmt_skip)]
    let tbl = [
        //  exp2(z + eps)                   eps
        f64::from_bits(0x3fe6a09e667f3d5d), f64::from_bits(0x3d39880000000000),
        f64::from_bits(0x3fe6b052fa751744), f64::from_bits(0x3cd8000000000000),
        f64::from_bits(0x3fe6c012750bd9fe), f64::from_bits(0xbd28780000000000),
        f64::from_bits(0x3fe6cfdcddd476bf), f64::from_bits(0x3d1ec00000000000),
        f64::from_bits(0x3fe6dfb23c651a29), f64::from_bits(0xbcd8000000000000),
        f64::from_bits(0x3fe6ef9298593ae3), f64::from_bits(0xbcbc000000000000),
        f64::from_bits(0x3fe6ff7df9519386), f64::from_bits(0xbd2fd80000000000),
        f64::from_bits(0x3fe70f7466f42da3), f64::from_bits(0xbd2c880000000000),
        f64::from_bits(0x3fe71f75e8ec5fc3), f64::from_bits(0x3d13c00000000000),
        f64::from_bits(0x3fe72f8286eacf05), f64::from_bits(0xbd38300000000000),
        f64::from_bits(0x3fe73f9a48a58152), f64::from_bits(0xbd00c00000000000),
        f64::from_bits(0x3fe74fbd35d7ccfc), f64::from_bits(0x3d2f880000000000),
        f64::from_bits(0x3fe75feb564267f1), f64::from_bits(0x3d03e00000000000),
        f64::from_bits(0x3fe77024b1ab6d48), f64::from_bits(0xbd27d00000000000),
        f64::from_bits(0x3fe780694fde5d38), f64::from_bits(0xbcdd000000000000),
        f64::from_bits(0x3fe790b938ac1d00), f64::from_bits(0x3ce3000000000000),
        f64::from_bits(0x3fe7a11473eb0178), f64::from_bits(0xbced000000000000),
        f64::from_bits(0x3fe7b17b0976d060), f64::from_bits(0x3d20400000000000),
        f64::from_bits(0x3fe7c1ed0130c133), f64::from_bits(0x3ca0000000000000),
        f64::from_bits(0x3fe7d26a62ff8636), f64::from_bits(0xbd26900000000000),
        f64::from_bits(0x3fe7e2f336cf4e3b), f64::from_bits(0xbd02e00000000000),
        f64::from_bits(0x3fe7f3878491c3e8), f64::from_bits(0xbd24580000000000),
        f64::from_bits(0x3fe80427543e1b4e), f64::from_bits(0x3d33000000000000),
        f64::from_bits(0x3fe814d2add1071a), f64::from_bits(0x3d0f000000000000),
        f64::from_bits(0x3fe82589994ccd7e), f64::from_bits(0xbd21c00000000000),
        f64::from_bits(0x3fe8364c1eb942d0), f64::from_bits(0x3d29d00000000000),
        f64::from_bits(0x3fe8471a4623cab5), f64::from_bits(0x3d47100000000000),
        f64::from_bits(0x3fe857f4179f5bbc), f64::from_bits(0x3d22600000000000),
        f64::from_bits(0x3fe868d99b4491af), f64::from_bits(0xbd32c40000000000),
        f64::from_bits(0x3fe879cad931a395), f64::from_bits(0xbd23000000000000),
        f64::from_bits(0x3fe88ac7d98a65b8), f64::from_bits(0xbd2a800000000000),
        f64::from_bits(0x3fe89bd0a4785800), f64::from_bits(0xbced000000000000),
        f64::from_bits(0x3fe8ace5422aa223), f64::from_bits(0x3d33280000000000),
        f64::from_bits(0x3fe8be05bad619fa), f64::from_bits(0x3d42b40000000000),
        f64::from_bits(0x3fe8cf3216b54383), f64::from_bits(0xbd2ed00000000000),
        f64::from_bits(0x3fe8e06a5e08664c), f64::from_bits(0xbd20500000000000),
        f64::from_bits(0x3fe8f1ae99157807), f64::from_bits(0x3d28280000000000),
        f64::from_bits(0x3fe902fed0282c0e), f64::from_bits(0xbd1cb00000000000),
        f64::from_bits(0x3fe9145b0b91ff96), f64::from_bits(0xbd05e00000000000),
        f64::from_bits(0x3fe925c353aa2ff9), f64::from_bits(0x3cf5400000000000),
        f64::from_bits(0x3fe93737b0cdc64a), f64::from_bits(0x3d17200000000000),
        f64::from_bits(0x3fe948b82b5f98ae), f64::from_bits(0xbd09000000000000),
        f64::from_bits(0x3fe95a44cbc852cb), f64::from_bits(0x3d25680000000000),
        f64::from_bits(0x3fe96bdd9a766f21), f64::from_bits(0xbd36d00000000000),
        f64::from_bits(0x3fe97d829fde4e2a), f64::from_bits(0xbd01000000000000),
        f64::from_bits(0x3fe98f33e47a23a3), f64::from_bits(0x3d2d000000000000),
        f64::from_bits(0x3fe9a0f170ca0604), f64::from_bits(0xbd38a40000000000),
        f64::from_bits(0x3fe9b2bb4d53ff89), f64::from_bits(0x3d355c0000000000),
        f64::from_bits(0x3fe9c49182a3f15b), f64::from_bits(0x3d26b80000000000),
        f64::from_bits(0x3fe9d674194bb8c5), f64::from_bits(0xbcec000000000000),
        f64::from_bits(0x3fe9e86319e3238e), f64::from_bits(0x3d17d00000000000),
        f64::from_bits(0x3fe9fa5e8d07f302), f64::from_bits(0x3d16400000000000),
        f64::from_bits(0x3fea0c667b5de54d), f64::from_bits(0xbcf5000000000000),
        f64::from_bits(0x3fea1e7aed8eb8f6), f64::from_bits(0x3d09e00000000000),
        f64::from_bits(0x3fea309bec4a2e27), f64::from_bits(0x3d2ad80000000000),
        f64::from_bits(0x3fea42c980460a5d), f64::from_bits(0xbd1af00000000000),
        f64::from_bits(0x3fea5503b23e259b), f64::from_bits(0x3d0b600000000000),
        f64::from_bits(0x3fea674a8af46213), f64::from_bits(0x3d38880000000000),
        f64::from_bits(0x3fea799e1330b3a7), f64::from_bits(0x3d11200000000000),
        f64::from_bits(0x3fea8bfe53c12e8d), f64::from_bits(0x3d06c00000000000),
        f64::from_bits(0x3fea9e6b5579fcd2), f64::from_bits(0xbd29b80000000000),
        f64::from_bits(0x3feab0e521356fb8), f64::from_bits(0x3d2b700000000000),
        f64::from_bits(0x3feac36bbfd3f381), f64::from_bits(0x3cd9000000000000),
        f64::from_bits(0x3fead5ff3a3c2780), f64::from_bits(0x3ce4000000000000),
        f64::from_bits(0x3feae89f995ad2a3), f64::from_bits(0xbd2c900000000000),
        f64::from_bits(0x3feafb4ce622f367), f64::from_bits(0x3d16500000000000),
        f64::from_bits(0x3feb0e07298db790), f64::from_bits(0x3d2fd40000000000),
        f64::from_bits(0x3feb20ce6c9a89a9), f64::from_bits(0x3d12700000000000),
        f64::from_bits(0x3feb33a2b84f1a4b), f64::from_bits(0x3d4d470000000000),
        f64::from_bits(0x3feb468415b747e7), f64::from_bits(0xbd38380000000000),
        f64::from_bits(0x3feb59728de5593a), f64::from_bits(0x3c98000000000000),
        f64::from_bits(0x3feb6c6e29f1c56a), f64::from_bits(0x3d0ad00000000000),
        f64::from_bits(0x3feb7f76f2fb5e50), f64::from_bits(0x3cde800000000000),
        f64::from_bits(0x3feb928cf22749b2), f64::from_bits(0xbd04c00000000000),
        f64::from_bits(0x3feba5b030a10603), f64::from_bits(0xbd0d700000000000),
        f64::from_bits(0x3febb8e0b79a6f66), f64::from_bits(0x3d0d900000000000),
        f64::from_bits(0x3febcc1e904bc1ff), f64::from_bits(0x3d02a00000000000),
        f64::from_bits(0x3febdf69c3f3a16f), f64::from_bits(0xbd1f780000000000),
        f64::from_bits(0x3febf2c25bd71db8), f64::from_bits(0xbd10a00000000000),
        f64::from_bits(0x3fec06286141b2e9), f64::from_bits(0xbd11400000000000),
        f64::from_bits(0x3fec199bdd8552e0), f64::from_bits(0x3d0be00000000000),
        f64::from_bits(0x3fec2d1cd9fa64ee), f64::from_bits(0xbd09400000000000),
        f64::from_bits(0x3fec40ab5fffd02f), f64::from_bits(0xbd0ed00000000000),
        f64::from_bits(0x3fec544778fafd15), f64::from_bits(0x3d39660000000000),
        f64::from_bits(0x3fec67f12e57d0cb), f64::from_bits(0xbd1a100000000000),
        f64::from_bits(0x3fec7ba88988c1b6), f64::from_bits(0xbd58458000000000),
        f64::from_bits(0x3fec8f6d9406e733), f64::from_bits(0xbd1a480000000000),
        f64::from_bits(0x3feca3405751c4df), f64::from_bits(0x3ccb000000000000),
        f64::from_bits(0x3fecb720dcef9094), f64::from_bits(0x3d01400000000000),
        f64::from_bits(0x3feccb0f2e6d1689), f64::from_bits(0x3cf0200000000000),
        f64::from_bits(0x3fecdf0b555dc412), f64::from_bits(0x3cf3600000000000),
        f64::from_bits(0x3fecf3155b5bab3b), f64::from_bits(0xbd06900000000000),
        f64::from_bits(0x3fed072d4a0789bc), f64::from_bits(0x3d09a00000000000),
        f64::from_bits(0x3fed1b532b08c8fa), f64::from_bits(0xbd15e00000000000),
        f64::from_bits(0x3fed2f87080d8a85), f64::from_bits(0x3d1d280000000000),
        f64::from_bits(0x3fed43c8eacaa203), f64::from_bits(0x3d01a00000000000),
        f64::from_bits(0x3fed5818dcfba491), f64::from_bits(0x3cdf000000000000),
        f64::from_bits(0x3fed6c76e862e6a1), f64::from_bits(0xbd03a00000000000),
        f64::from_bits(0x3fed80e316c9834e), f64::from_bits(0xbd0cd80000000000),
        f64::from_bits(0x3fed955d71ff6090), f64::from_bits(0x3cf4c00000000000),
        f64::from_bits(0x3feda9e603db32ae), f64::from_bits(0x3cff900000000000),
        f64::from_bits(0x3fedbe7cd63a8325), f64::from_bits(0x3ce9800000000000),
        f64::from_bits(0x3fedd321f301b445), f64::from_bits(0xbcf5200000000000),
        f64::from_bits(0x3fede7d5641c05bf), f64::from_bits(0xbd1d700000000000),
        f64::from_bits(0x3fedfc97337b9aec), f64::from_bits(0xbd16140000000000),
        f64::from_bits(0x3fee11676b197d5e), f64::from_bits(0x3d0b480000000000),
        f64::from_bits(0x3fee264614f5a3e7), f64::from_bits(0x3d40ce0000000000),
        f64::from_bits(0x3fee3b333b16ee5c), f64::from_bits(0x3d0c680000000000),
        f64::from_bits(0x3fee502ee78b3fb4), f64::from_bits(0xbd09300000000000),
        f64::from_bits(0x3fee653924676d68), f64::from_bits(0xbce5000000000000),
        f64::from_bits(0x3fee7a51fbc74c44), f64::from_bits(0xbd07f80000000000),
        f64::from_bits(0x3fee8f7977cdb726), f64::from_bits(0xbcf3700000000000),
        f64::from_bits(0x3feea4afa2a490e8), f64::from_bits(0x3ce5d00000000000),
        f64::from_bits(0x3feeb9f4867ccae4), f64::from_bits(0x3d161a0000000000),
        f64::from_bits(0x3feecf482d8e680d), f64::from_bits(0x3cf5500000000000),
        f64::from_bits(0x3feee4aaa2188514), f64::from_bits(0x3cc6400000000000),
        f64::from_bits(0x3feefa1bee615a13), f64::from_bits(0xbcee800000000000),
        f64::from_bits(0x3fef0f9c1cb64106), f64::from_bits(0xbcfa880000000000),
        f64::from_bits(0x3fef252b376bb963), f64::from_bits(0xbd2c900000000000),
        f64::from_bits(0x3fef3ac948dd7275), f64::from_bits(0x3caa000000000000),
        f64::from_bits(0x3fef50765b6e4524), f64::from_bits(0xbcf4f00000000000),
        f64::from_bits(0x3fef6632798844fd), f64::from_bits(0x3cca800000000000),
        f64::from_bits(0x3fef7bfdad9cbe38), f64::from_bits(0x3cfabc0000000000),
        f64::from_bits(0x3fef91d802243c82), f64::from_bits(0xbcd4600000000000),
        f64::from_bits(0x3fefa7c1819e908e), f64::from_bits(0xbd0b0c0000000000),
        f64::from_bits(0x3fefbdba3692d511), f64::from_bits(0xbcc0e00000000000),
        f64::from_bits(0x3fefd3c22b8f7194), f64::from_bits(0xbd10de8000000000),
        f64::from_bits(0x3fefe9d96b2a23ee), f64::from_bits(0x3cee430000000000),
        f64::from_bits(0x3ff0000000000000), f64::from_bits(0x0),
        f64::from_bits(0x3ff00b1afa5abcbe), f64::from_bits(0xbcb3400000000000),
        f64::from_bits(0x3ff0163da9fb3303), f64::from_bits(0xbd12170000000000),
        f64::from_bits(0x3ff02168143b0282), f64::from_bits(0x3cba400000000000),
        f64::from_bits(0x3ff02c9a3e77806c), f64::from_bits(0x3cef980000000000),
        f64::from_bits(0x3ff037d42e11bbca), f64::from_bits(0xbcc7400000000000),
        f64::from_bits(0x3ff04315e86e7f89), f64::from_bits(0x3cd8300000000000),
        f64::from_bits(0x3ff04e5f72f65467), f64::from_bits(0xbd1a3f0000000000),
        f64::from_bits(0x3ff059b0d315855a), f64::from_bits(0xbd02840000000000),
        f64::from_bits(0x3ff0650a0e3c1f95), f64::from_bits(0x3cf1600000000000),
        f64::from_bits(0x3ff0706b29ddf71a), f64::from_bits(0x3d15240000000000),
        f64::from_bits(0x3ff07bd42b72a82d), f64::from_bits(0xbce9a00000000000),
        f64::from_bits(0x3ff0874518759bd0), f64::from_bits(0x3ce6400000000000),
        f64::from_bits(0x3ff092bdf66607c8), f64::from_bits(0xbd00780000000000),
        f64::from_bits(0x3ff09e3ecac6f383), f64::from_bits(0xbc98000000000000),
        f64::from_bits(0x3ff0a9c79b1f3930), f64::from_bits(0x3cffa00000000000),
        f64::from_bits(0x3ff0b5586cf988fc), f64::from_bits(0xbcfac80000000000),
        f64::from_bits(0x3ff0c0f145e46c8a), f64::from_bits(0x3cd9c00000000000),
        f64::from_bits(0x3ff0cc922b724816), f64::from_bits(0x3d05200000000000),
        f64::from_bits(0x3ff0d83b23395dd8), f64::from_bits(0xbcfad00000000000),
        f64::from_bits(0x3ff0e3ec32d3d1f3), f64::from_bits(0x3d1bac0000000000),
        f64::from_bits(0x3ff0efa55fdfa9a6), f64::from_bits(0xbd04e80000000000),
        f64::from_bits(0x3ff0fb66affed2f0), f64::from_bits(0xbd0d300000000000),
        f64::from_bits(0x3ff1073028d7234b), f64::from_bits(0x3cf1500000000000),
        f64::from_bits(0x3ff11301d0125b5b), f64::from_bits(0x3cec000000000000),
        f64::from_bits(0x3ff11edbab5e2af9), f64::from_bits(0x3d16bc0000000000),
        f64::from_bits(0x3ff12abdc06c31d5), f64::from_bits(0x3ce8400000000000),
        f64::from_bits(0x3ff136a814f2047d), f64::from_bits(0xbd0ed00000000000),
        f64::from_bits(0x3ff1429aaea92de9), f64::from_bits(0x3ce8e00000000000),
        f64::from_bits(0x3ff14e95934f3138), f64::from_bits(0x3ceb400000000000),
        f64::from_bits(0x3ff15a98c8a58e71), f64::from_bits(0x3d05300000000000),
        f64::from_bits(0x3ff166a45471c3df), f64::from_bits(0x3d03380000000000),
        f64::from_bits(0x3ff172b83c7d5211), f64::from_bits(0x3d28d40000000000),
        f64::from_bits(0x3ff17ed48695bb9f), f64::from_bits(0xbd05d00000000000),
        f64::from_bits(0x3ff18af9388c8d93), f64::from_bits(0xbd1c880000000000),
        f64::from_bits(0x3ff1972658375d66), f64::from_bits(0x3d11f00000000000),
        f64::from_bits(0x3ff1a35beb6fcba7), f64::from_bits(0x3d10480000000000),
        f64::from_bits(0x3ff1af99f81387e3), f64::from_bits(0xbd47390000000000),
        f64::from_bits(0x3ff1bbe084045d54), f64::from_bits(0x3d24e40000000000),
        f64::from_bits(0x3ff1c82f95281c43), f64::from_bits(0xbd0a200000000000),
        f64::from_bits(0x3ff1d4873168b9b2), f64::from_bits(0x3ce3800000000000),
        f64::from_bits(0x3ff1e0e75eb44031), f64::from_bits(0x3ceac00000000000),
        f64::from_bits(0x3ff1ed5022fcd938), f64::from_bits(0x3d01900000000000),
        f64::from_bits(0x3ff1f9c18438cdf7), f64::from_bits(0xbd1b780000000000),
        f64::from_bits(0x3ff2063b88628d8f), f64::from_bits(0x3d2d940000000000),
        f64::from_bits(0x3ff212be3578a81e), f64::from_bits(0x3cd8000000000000),
        f64::from_bits(0x3ff21f49917ddd41), f64::from_bits(0x3d2b340000000000),
        f64::from_bits(0x3ff22bdda2791323), f64::from_bits(0x3d19f80000000000),
        f64::from_bits(0x3ff2387a6e7561e7), f64::from_bits(0xbd19c80000000000),
        f64::from_bits(0x3ff2451ffb821427), f64::from_bits(0x3d02300000000000),
        f64::from_bits(0x3ff251ce4fb2a602), f64::from_bits(0xbd13480000000000),
        f64::from_bits(0x3ff25e85711eceb0), f64::from_bits(0x3d12700000000000),
        f64::from_bits(0x3ff26b4565e27d16), f64::from_bits(0x3d11d00000000000),
        f64::from_bits(0x3ff2780e341de00f), f64::from_bits(0x3d31ee0000000000),
        f64::from_bits(0x3ff284dfe1f5633e), f64::from_bits(0xbd14c00000000000),
        f64::from_bits(0x3ff291ba7591bb30), f64::from_bits(0xbd13d80000000000),
        f64::from_bits(0x3ff29e9df51fdf09), f64::from_bits(0x3d08b00000000000),
        f64::from_bits(0x3ff2ab8a66d10e9b), f64::from_bits(0xbd227c0000000000),
        f64::from_bits(0x3ff2b87fd0dada3a), f64::from_bits(0x3d2a340000000000),
        f64::from_bits(0x3ff2c57e39771af9), f64::from_bits(0xbd10800000000000),
        f64::from_bits(0x3ff2d285a6e402d9), f64::from_bits(0xbd0ed00000000000),
        f64::from_bits(0x3ff2df961f641579), f64::from_bits(0xbcf4200000000000),
        f64::from_bits(0x3ff2ecafa93e2ecf), f64::from_bits(0xbd24980000000000),
        f64::from_bits(0x3ff2f9d24abd8822), f64::from_bits(0xbd16300000000000),
        f64::from_bits(0x3ff306fe0a31b625), f64::from_bits(0xbd32360000000000),
        f64::from_bits(0x3ff31432edeea50b), f64::from_bits(0xbd70df8000000000),
        f64::from_bits(0x3ff32170fc4cd7b8), f64::from_bits(0xbd22480000000000),
        f64::from_bits(0x3ff32eb83ba8e9a2), f64::from_bits(0xbd25980000000000),
        f64::from_bits(0x3ff33c08b2641766), f64::from_bits(0x3d1ed00000000000),
        f64::from_bits(0x3ff3496266e3fa27), f64::from_bits(0xbcdc000000000000),
        f64::from_bits(0x3ff356c55f929f0f), f64::from_bits(0xbd30d80000000000),
        f64::from_bits(0x3ff36431a2de88b9), f64::from_bits(0x3d22c80000000000),
        f64::from_bits(0x3ff371a7373aaa39), f64::from_bits(0x3d20600000000000),
        f64::from_bits(0x3ff37f26231e74fe), f64::from_bits(0xbd16600000000000),
        f64::from_bits(0x3ff38cae6d05d838), f64::from_bits(0xbd0ae00000000000),
        f64::from_bits(0x3ff39a401b713ec3), f64::from_bits(0xbd44720000000000),
        f64::from_bits(0x3ff3a7db34e5a020), f64::from_bits(0x3d08200000000000),
        f64::from_bits(0x3ff3b57fbfec6e95), f64::from_bits(0x3d3e800000000000),
        f64::from_bits(0x3ff3c32dc313a8f2), f64::from_bits(0x3cef800000000000),
        f64::from_bits(0x3ff3d0e544ede122), f64::from_bits(0xbd17a00000000000),
        f64::from_bits(0x3ff3dea64c1234bb), f64::from_bits(0x3d26300000000000),
        f64::from_bits(0x3ff3ec70df1c4ecc), f64::from_bits(0xbd48a60000000000),
        f64::from_bits(0x3ff3fa4504ac7e8c), f64::from_bits(0xbd3cdc0000000000),
        f64::from_bits(0x3ff40822c367a0bb), f64::from_bits(0x3d25b80000000000),
        f64::from_bits(0x3ff4160a21f72e95), f64::from_bits(0x3d1ec00000000000),
        f64::from_bits(0x3ff423fb27094646), f64::from_bits(0xbd13600000000000),
        f64::from_bits(0x3ff431f5d950a920), f64::from_bits(0x3d23980000000000),
        f64::from_bits(0x3ff43ffa3f84b9eb), f64::from_bits(0x3cfa000000000000),
        f64::from_bits(0x3ff44e0860618919), f64::from_bits(0xbcf6c00000000000),
        f64::from_bits(0x3ff45c2042a7d201), f64::from_bits(0xbd0bc00000000000),
        f64::from_bits(0x3ff46a41ed1d0016), f64::from_bits(0xbd12800000000000),
        f64::from_bits(0x3ff4786d668b3326), f64::from_bits(0x3d30e00000000000),
        f64::from_bits(0x3ff486a2b5c13c00), f64::from_bits(0xbd2d400000000000),
        f64::from_bits(0x3ff494e1e192af04), f64::from_bits(0x3d0c200000000000),
        f64::from_bits(0x3ff4a32af0d7d372), f64::from_bits(0xbd1e500000000000),
        f64::from_bits(0x3ff4b17dea6db801), f64::from_bits(0x3d07800000000000),
        f64::from_bits(0x3ff4bfdad53629e1), f64::from_bits(0xbd13800000000000),
        f64::from_bits(0x3ff4ce41b817c132), f64::from_bits(0x3d00800000000000),
        f64::from_bits(0x3ff4dcb299fddddb), f64::from_bits(0x3d2c700000000000),
        f64::from_bits(0x3ff4eb2d81d8ab96), f64::from_bits(0xbd1ce00000000000),
        f64::from_bits(0x3ff4f9b2769d2d02), f64::from_bits(0x3d19200000000000),
        f64::from_bits(0x3ff508417f4531c1), f64::from_bits(0xbd08c00000000000),
        f64::from_bits(0x3ff516daa2cf662a), f64::from_bits(0xbcfa000000000000),
        f64::from_bits(0x3ff5257de83f51ea), f64::from_bits(0x3d4a080000000000),
        f64::from_bits(0x3ff5342b569d4eda), f64::from_bits(0xbd26d80000000000),
        f64::from_bits(0x3ff542e2f4f6ac1a), f64::from_bits(0xbd32440000000000),
        f64::from_bits(0x3ff551a4ca5d94db), f64::from_bits(0x3d483c0000000000),
        f64::from_bits(0x3ff56070dde9116b), f64::from_bits(0x3d24b00000000000),
        f64::from_bits(0x3ff56f4736b529de), f64::from_bits(0x3d415a0000000000),
        f64::from_bits(0x3ff57e27dbe2c40e), f64::from_bits(0xbd29e00000000000),
        f64::from_bits(0x3ff58d12d497c76f), f64::from_bits(0xbd23080000000000),
        f64::from_bits(0x3ff59c0827ff0b4c), f64::from_bits(0x3d4dec0000000000),
        f64::from_bits(0x3ff5ab07dd485427), f64::from_bits(0xbcc4000000000000),
        f64::from_bits(0x3ff5ba11fba87af4), f64::from_bits(0x3d30080000000000),
        f64::from_bits(0x3ff5c9268a59460b), f64::from_bits(0xbd26c80000000000),
        f64::from_bits(0x3ff5d84590998e3f), f64::from_bits(0x3d469a0000000000),
        f64::from_bits(0x3ff5e76f15ad20e1), f64::from_bits(0xbd1b400000000000),
        f64::from_bits(0x3ff5f6a320dcebca), f64::from_bits(0x3d17700000000000),
        f64::from_bits(0x3ff605e1b976dcb8), f64::from_bits(0x3d26f80000000000),
        f64::from_bits(0x3ff6152ae6cdf715), f64::from_bits(0x3d01000000000000),
        f64::from_bits(0x3ff6247eb03a5531), f64::from_bits(0xbd15d00000000000),
        f64::from_bits(0x3ff633dd1d1929b5), f64::from_bits(0xbd12d00000000000),
        f64::from_bits(0x3ff6434634ccc313), f64::from_bits(0xbcea800000000000),
        f64::from_bits(0x3ff652b9febc8efa), f64::from_bits(0xbd28600000000000),
        f64::from_bits(0x3ff6623882553397), f64::from_bits(0x3d71fe0000000000),
        f64::from_bits(0x3ff671c1c708328e), f64::from_bits(0xbd37200000000000),
        f64::from_bits(0x3ff68155d44ca97e), f64::from_bits(0x3ce6800000000000),
        f64::from_bits(0x3ff690f4b19e9471), f64::from_bits(0xbd29780000000000),
    ];

    // double_t r, t, z;
    // uint32_t ix, i0;
    // union {double f; uint64_t i;} u = {x};
    // union {uint32_t u; int32_t i;} k;
    let x1p1023 = f64::from_bits(0x7fe0000000000000);
    let x1p52 = f64::from_bits(0x4330000000000000);
    let _0x1p_149 = f64::from_bits(0xb6a0000000000000);

    /* Filter out exceptional cases. */
    let ui = f64::to_bits(x);
    let ix = ui >> 32 & 0x7fffffff;
    if ix >= 0x408ff000 {
        /* |x| >= 1022 or nan */
        if ix >= 0x40900000 && ui >> 63 == 0 {
            /* x >= 1024 or nan */
            /* overflow */
            x *= x1p1023;
            return x;
        }
        if ix >= 0x7ff00000 {
            /* -inf or -nan */
            return -1.0 / x;
        }
        if ui >> 63 != 0 {
            /* x <= -1022 */
            /* underflow */
            if x <= -1075.0 || x - x1p52 + x1p52 != x {
                force_eval!((_0x1p_149 / x) as f32);
            }
            if x <= -1075.0 {
                return 0.0;
            }
        }
    } else if ix < 0x3c900000 {
        /* |x| < 0x1p-54 */
        return 1.0 + x;
    }

    /* Reduce x, computing z, i0, and k. */
    let ui = f64::to_bits(x + redux);
    let mut i0 = ui as u32;
    i0 += TBLSIZE as u32 / 2;
    let ku = i0 / TBLSIZE as u32 * TBLSIZE as u32;
    let ki = ku as i32 / TBLSIZE as i32;
    i0 %= TBLSIZE as u32;
    let uf = f64::from_bits(ui) - redux;
    let mut z = x - uf;

    /* Compute r = exp2(y) = exp2t[i0] * p(z - eps[i]). */
    let t = tbl[2 * i0 as usize]; /* exp2t[i0] */
    z -= tbl[2 * i0 as usize + 1]; /* eps[i0]   */
    let r = t + t * z * (p1 + z * (p2 + z * (p3 + z * (p4 + z * p5))));

    scalbn(r, ki)
}

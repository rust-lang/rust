This program emits `psllw` instead of `pslliw` instruction. `pslliw` should be
usable here since `imm8` is a constant:

```rust
extern crate stdsimd;

use std::env;
use stdsimd as s;

fn main() {
    let imm8: i32 = env::args().nth(1).unwrap().parse().unwrap();
    let a = s::i16x8::new(imm8 as i16, 0, 0, 0, 0, 0, 0, 0);
    println!("{:?}", s::_mm_slli_epi16(a, 4));
}
```

---

LLVM types the `count` parameter in `_mm_sll_epi16` as `i16x8`, but the right
type is probably `i64x2`.

---

What is the deal with the ucomi f64 comparison functions in SSE2? Clang's
headers indicate some specific behavior with NAN, but I can't seem to reproduce
it. Intel's official docs are very vague.

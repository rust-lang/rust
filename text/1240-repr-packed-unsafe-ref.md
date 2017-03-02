- Feature Name: NA
- Start Date: 2015-08-06
- RFC PR: https://github.com/rust-lang/rfcs/pull/1240
- Rust Issue: https://github.com/rust-lang/rust/issues/27060

# Summary

Taking a reference into a struct marked `repr(packed)` should become
`unsafe`, because it can lead to undefined behaviour. `repr(packed)`
structs need to be banned from storing `Drop` types for this reason.

# Motivation

Issue [#27060](https://github.com/rust-lang/rust/issues/27060) noticed
that it was possible to trigger undefined behaviour in safe code via
`repr(packed)`, by creating references `&T` which don't satisfy the
expected alignment requirements for `T`.

Concretely, the compiler assumes that any reference (or raw pointer,
in fact) will be aligned to at least `align_of::<T>()`, i.e. the
following snippet should run successfully:

```rust
let some_reference: &T = /* arbitrary code */;

let actual_address = some_reference as *const _ as usize;
let align = std::mem::align_of::<T>();

assert_eq!(actual_address % align, 0);
```

However, `repr(packed)` allows on to violate this, by creating values
of arbitrary types that are stored at "random" byte addresses, by
removing the padding normally inserted to maintain alignment in
`struct`s. E.g. suppose there's a struct `Foo` defined like
`#[repr(packed, C)] struct Foo { x: u8, y: u32 }`, and there's an
instance of `Foo` allocated at a 0x1000, the `u32` will be placed at
`0x1001`, which isn't 4-byte aligned (the alignment of `u32`).

Issue #27060 has a snippet which crashes at runtime on at least two
x86-64 CPUs (the author's and the one playpen runs on) and almost
certainly most other platforms.

```rust
#![feature(simd, test)]

extern crate test;

// simd types require high alignment or the CPU faults
#[simd]
#[derive(Debug, Copy, Clone)]
struct f32x4(f32, f32, f32, f32);

#[repr(packed)]
#[derive(Copy, Clone)]
struct Unalign<T>(T);

struct Breakit {
    x: u8,
    y: Unalign<f32x4>
}

fn main() {
    let val = Breakit { x: 0, y: Unalign(f32x4(0.0, 0.0, 0.0, 0.0)) };

    test::black_box(&val);

    println!("before");

    let ok = val.y;
    test::black_box(ok.0);

    println!("middle");

    let bad = val.y.0;
    test::black_box(bad);

    println!("after");
}
```

On playpen, it prints:

```
before
middle
playpen: application terminated abnormally with signal 4 (Illegal instruction)
```

That is, the `bad` variable is causing the CPU to fault. The `let`
statement is (in pseudo-Rust) behaving like `let bad =
load_with_alignment(&val.y.0, align_of::<f32x4>());`, but the
alignment isn't satisfied. (The `ok` line is compiled to a `movupd`
instruction, while the `bad` is compiled to a `movapd`: `u` ==
unaligned, `a` == aligned.)

(NB. The use of SIMD types in the example is just to be able to
demonstrate the problem on x86. That platform is generally fairly
relaxed about pointer alignments and so SIMD & its specialised `mov`
instructions are the easiest way to demonstrate the violated
assumptions at runtime. Other platforms may fault on other types.)

Being able to assume that accesses are aligned is useful, for
performance, and almost all references will be correctly aligned
anyway (`repr(packed)` types and internal references into them are
quite rare).

The problems with unaligned accesses can be avoided by ensuring that
the accesses are actually aligned (e.g. via runtime checks, or other
external constraints the compiler cannot understand directly). For
example, consider the following

```rust
#[repr(packed, C)]
struct Bar {
    x: u8,
    y: u16,
    z: u8,
    w: u32,
}
```

Taking a reference to some of those fields may cause undefined
behaviour, but not always. It is always correct to take
a reference to `x` or `z` since `u8` has alignment 1. If the struct
value itself is 4-byte aligned (which is not guaranteed), `w` will
also be 4-byte aligned since the `u8, u16, u8` take up 4 bytes, hence
it is correct to take a reference to `w` in this case (and only that
case). Similarly, it is only correct to take a reference to `y` if the
struct is at an odd address, so that the `u16` starts at an even one
(i.e. is 2-byte aligned).

# Detailed design

It is `unsafe` to take a reference to the field of a `repr(packed)`
struct. It is still possible, but it is up to the programmer to ensure
that the alignment requirements are satisfied. Referencing
(by-reference, or by-value) a subfield of a struct (including indexing
elements of a fixed-length array) stored inside a `repr(packed)`
struct counts as taking a reference to the `packed` field and hence is
unsafe.

It is still legal to manipulate the fields of a `packed` struct by
value, e.g. the following is correct (and not `unsafe`), no matter the
alignment of `bar`:

```rust
let bar: Bar = ...;

let x = bar.y;
bar.w = 10;
```

It is illegal to store a type `T` implementing `Drop` (including a
generic type) in a `repr(packed)` type, since the destructor of `T` is
passed a reference to that `T`. The crater run (see appendix) found no
crate that needs to use `repr(packed)` to store a `Drop` type (or a
generic type). The generic type rule is conservatively approximated by
disallowing generic `repr(packed)` structs altogether, but this can be
relaxed (see Alternatives).

Concretely, this RFC is proposing the introduction of the `// error`s
in the following code.

```rust
struct Baz {
    x: u8,
}

#[repr(packed)]
struct Qux<T> { // error: generic repr(packed) struct
    y: Baz,
    z: u8,
    w: String, // error: storing a Drop type in a repr(packed) struct
    t: [u8; 4],
}

let mut qux = Qux { ... };

// all ok:
let y_val = qux.y;
let z_val = qux.z;
let t_val = qux.t;
qux.y = Baz { ... };
qux.z = 10;
qux.t = [0, 1, 2, 3];

// new errors:

let y_ref = &qux.y; // error: taking a reference to a field of a repr(packed) struct is unsafe
let z_ref = &mut qux.z; // ditto
let y_ptr: *const _ = &qux.y; // ditto
let z_ptr: *mut _ = &mut qux.z; // ditto

let x_val = qux.y.x; // error: directly using a subfield of a field of a repr(packed) struct is unsafe
let x_ref = &qux.y.x; // ditto
qux.y.x = 10; // ditto

let t_val = qux.t[0]; // error: directly indexing an array in a field of a repr(packed) struct is unsafe
let t_ref = &qux.t[0]; // ditto
qux.t[0] = 10; // ditto
```

(NB. the subfield and indexing cases can be resolved by first copying
the packed field's value onto the stack, and then accessing the
desired value.)

## Staging

This change will first land as warnings indicating that code will be
broken, with the warnings switched to the intended errors after one
release cycle.

# Drawbacks

This will cause some functionality to stop working in
possibly-surprising ways (NB. the drawback here is mainly the
"possibly-surprising", since the functionality is broken with general
`packed` types.). For example, `#[derive]` usually takes references to
the fields of structs, and so `#[derive(Clone)]` will generate
errors. However, this use of derive is incorrect in general (no
guarantee that the fields are aligned), and, one can easily replace it
by:

```rust
#[derive(Copy)]
#[repr(packed)]
struct Foo { ... }

impl Clone for Foo { fn clone(&self) -> Foo { *self } }
```

Similarly, `println!("{}", foo.bar)` will be an error despite there
not being a visible reference (`println!` takes one internally),
however, this can be resolved by, for instance, assigning to a
temporary.

# Alternatives

- A short-term solution would be to feature gate `repr(packed)` while
  the kinks are worked out of it
- Taking an internal reference could be made flat-out illegal, and the
  times when it is correct simulated by manual raw-pointer
  manipulation.
- The rules could be made less conservative in several cases, however
  the crater run didn't indicate any need for this:
  - a generic `repr(packed)` struct can use the generic in ways that
    avoids problems with `Drop`, e.g. if the generic is bounded by
    `Copy`, or if the type is only used in ways that are `Copy` such
    as behind a `*const T`.
  - using a subfield of a field of a `repr(packed)` struct by-value
    could be OK.

# Unresolved questions

None.

# Appendix

## Crater analysis

Crater was run on 2015/07/23 with a patch that feature gated `repr(packed)`.

High-level summary:

- several unnecessary uses of `repr(packed)` (patches have been
  submitted and merged to remove all of these)
- most necessary ones are to match the declaration of a struct in C
- many "necessary" uses can be replaced by byte arrays/arrays of smaller types
- 8 crates are currently on stable themselves (unsure about deps), 4 are already on nightly
  - 1 of the 8, http2parse, is essentially only used by a nightly-only crate (tendril)
  - 4 of the stable and 1 of the nightly crates don't need `repr(packed)` at all

|            | stable | needed | FFI only |
|------------|--------|--------|----------|
| image      | ✓      |        |          |
| nix        | ✓      | ✓      | ✓        |
| tendril    |        | ✓      |          |
| assimp-sys | ✓      | ✓      | ✓        |
| stemmer    | ✓      |        |          |
| x86        | ✓      | ✓      | ✓        |
| http2parse | ✓      | ✓      |          |
| nl80211rs  | ✓      | ✓      | ✓        |
| openal     | ✓      |        |          |
| elfloader  |        | ✓      | ✓        |
| x11        | ✓      |        |          |
| kiss3d     | ✓      |        |          |

More detailed analysis inline with broken crates. (Don't miss `kiss3d` in the non-root section.)

### Regression report c85ba3e9cb4620c6ec8273a34cce6707e91778cb vs. 7a265c6d1280932ba1b881f31f04b03b20c258e5

* From: c85ba3e9cb4620c6ec8273a34cce6707e91778cb
* To: 7a265c6d1280932ba1b881f31f04b03b20c258e5

#### Coverage

* 2617 crates tested: 1404 working / 1151 broken / 40 regressed / 0 fixed / 22 unknown.

#### Regressions

* There are 11 root regressions
* There are 40 regressions

#### Root regressions, sorted by rank:

* [image-0.3.11](https://crates.io/crates/image)
  ([before](https://tools.taskcluster.net/task-inspector/#V6QBA9LfTT6mhFJ0Yo7nJg))
  ([after](https://tools.taskcluster.net/task-inspector/#QU9d4XEPSWOg7CIGFpATDg))
  - [use](https://github.com/PistonDevelopers/image/blob/8e64e0d78e465ddfa13cd6627dede5fd258386f6/src/tga/decoder.rs#L75)
    seems entirely unnecessary (no raw bytewise operations on the
    struct itself)

  On stable.
* [nix-0.3.9](https://crates.io/crates/nix)
  ([before](https://tools.taskcluster.net/task-inspector/#X3HMXrq4S_GMNbeeAY8i6w))
  ([after](https://tools.taskcluster.net/task-inspector/#kz0vDaAhRRuKww2l-FvYpQ))
  - [use](https://github.com/carllerche/nix-rust/blob/5801318c0c4c6eeb3431144a89496830f55d6628/src/sys/epoll.rs#L98)
    required to match
    [C struct](https://github.com/torvalds/linux/blob/de182468d1bb726198abaab315820542425270b7/include/uapi/linux/eventpoll.h#L53-L62)

  On stable.
* [tendril-0.1.2](https://crates.io/crates/tendril)
  ([before](https://tools.taskcluster.net/task-inspector/#zQH7ShADR5O9eQe1mg3e6A))
  ([after](https://tools.taskcluster.net/task-inspector/#zI-PoIZHTm-7Urq3CLsXeg))
  - [use 1](https://github.com/servo/tendril/blob/faf97ded26213e561f8ad2768113cc05b6424748/src/buf32.rs#L19)
    not strictly necessary?
  - [use 2](https://github.com/servo/tendril/blob/faf97ded26213e561f8ad2768113cc05b6424748/src/tendril.rs#L43)
    required on 64-bit platforms to get size_of::&lt;Header>() == 12 rather
    than 16.
  - [use 3](https://github.com/servo/tendril/blob/faf97ded26213e561f8ad2768113cc05b6424748/src/tendril.rs#L91),
  as above, does some precise tricks with the layout for optimisation.

  Requires nightly.
* [assimp-sys-0.0.3](https://crates.io/crates/assimp-sys) ([before](https://tools.taskcluster.net/task-inspector/#rTrUh0VQR2uWXMQw14kRIA)) ([after](https://tools.taskcluster.net/task-inspector/#AR36o35FRV-mVInHKWFDrg))
  - [many uses](https://github.com/Eljay/assimp-sys/search?utf8=%E2%9C%93&q=packed),
    required to match
    [C structs](https://github.com/assimp/assimp/blob/f3d418a199cfb7864c826665016e11c65ddd7aa9/include/assimp/types.h#L227)
    (one example). In author's words:

    > [11:36:15] &lt;eljay> huon: well my assimp binding is basically abandoned for now if you are just worried about breaking things, and seems unlikely anyone is using it :P

  On stable.
* [stemmer-0.1.1](https://crates.io/crates/stemmer) ([before](https://tools.taskcluster.net/task-inspector/#0Affr5PrTnGoBukeRwuiKw)) ([after](https://tools.taskcluster.net/task-inspector/#8xGRmPxOQS2NHbvgXMvmWQ))
  - [use](https://github.com/lady-segfault/stemmer-rs/blob/4090dcf7a258df5031c10754c8de118e0ca93512/src/stemmer.rs#L7), completely unnecessary

  On stable.
* [x86-0.2.0](https://crates.io/crates/x86) ([before](https://tools.taskcluster.net/task-inspector/#__VYVs6QSYm4JF68fSXibw)) ([after](https://tools.taskcluster.net/task-inspector/#xj8paeiaR0OGkK1v2raHYg))
  - [several similar uses](https://github.com/gz/rust-x86/search?utf8=%E2%9C%93&q=packed),
    specific layout necessary for raw interaction with CPU features

  Requires nightly.
* [http2parse-0.0.3](https://crates.io/crates/http2parse) ([before](https://tools.taskcluster.net/task-inspector/#CUr_5dfgQMywZmG_ER7ZGQ)) ([after](https://tools.taskcluster.net/task-inspector/#rQO3m_8iQQapN2l-PvGrRw))
  - [use](https://github.com/reem/rust-http2parse/blob/b363139ac2f81fa25db504a9256face9f8c799b6/src/payload.rs#L206),
    used to get super-fast "parsing" of headers, by transmuting
    `&[u8]` to `&[Setting]`.

  On stable, however:

    ```irc
    [11:30:38] <huon> reem: why is https://github.com/reem/rust-http2parse/blob/b363139ac2f81fa25db504a9256face9f8c799b6/src/payload.rs#L208 packed?
    [11:31:59] <reem> huon: I transmute from & [u8] to & [Setting]
    [11:32:35] <reem> So repr packed gets me the layout I need
    [11:32:47] <reem> With no padding between the u8 and u16
    [11:33:11] <reem> and between Settings
    [11:33:17] <huon> ok
    [11:33:22] <huon> (stop doing bad things :P )
    [11:34:00] <huon> (there's some problems with repr(packed) https://github.com/rust-lang/rust/issues/27060 and we may be feature gating it)
    [11:35:02] <huon> reem: wait, aren't there endianness problems?
    [11:36:16] <reem> Ah yes, looks like I forgot to finish the Setting interface
    [11:36:27] <reem> The identifier and value methods take care of converting to types values
    [11:36:39] <reem> The goal is just to avoid copying the whole buffer and requiring an allocation
    [11:37:01] <reem> Right now the whole parser takes like 9 ns to parse a frame
    [11:39:11] <huon> would you be sunk if repr(packed) was feature gated?
    [11:40:17] <huon> or, is maybe something like `struct SettingsRaw { identifier:  [u8; 2], value:  [u8; 4] }` OK (possibly with conversion functions etc.)?
    [11:40:46] <reem> Yea, I could get around it if I needed to
    [11:40:58] <reem> Anyway the primary consumer is transfer and I'm running on nightly there
    [11:41:05] <reem> So it doesn't matter too much
    ```

* [nl80211rs-0.1.0](https://crates.io/crates/nl80211rs) ([before](https://tools.taskcluster.net/task-inspector/#rhEG57vQQHWiVCcS3kIWrA)) ([after](https://tools.taskcluster.net/task-inspector/#s97ED8oXQ4WN-Pbm3ZsFJQ))
  - [three similar uses](https://github.com/carrotsrc/nl80211rs/search?utf8=%E2%9C%93&q=packed)
    to match
    [C struct](http://lxr.free-electrons.com/source/include/uapi/linux/nl80211.h#L2288).

  On stable.
* [openal-0.2.1](https://crates.io/crates/openal) ([before](https://tools.taskcluster.net/task-inspector/#XUvl-638T82xgGwkrxpz5g)) ([after](https://tools.taskcluster.net/task-inspector/#Oc9wEFpbQM2Tja9sv0qt4g))
  - [several similar uses](https://github.com/meh/rust-openal/blob/9e35fd284f25da7fe90a8307de85a6ec6d392ea1/src/util.rs#L6),
    probably unnecessary, just need the struct to behave like
    `[f32; 3]`: pointers to it
    [are passed](https://github.com/meh/rust-openal/blob/9e35fd284f25da7fe90a8307de85a6ec6d392ea1/src/listener/listener.rs#L204-L205)
    to [functions expecting `*mut f32`](https://github.com/meh/rust-openal-sys/blob/master/src/al.rs#L146) pointers.

  On stable.
* [elfloader-0.0.1](https://crates.io/crates/elfloader) ([before](https://tools.taskcluster.net/task-inspector/#ssE4lk0xR3q1qYZBXK24aA)) ([after](https://tools.taskcluster.net/task-inspector/#SAH7AAVIToKkhf7QRK4C1g))
  - [two similar uses](https://github.com/gz/rust-elfloader/blob/d61db7c83d66ce65da92aed5e33a4baf35f4c1e7/src/elf.rs#L362),
    required to match file headers/formats exactly.

  Requires nightly.
* [x11cap-0.1.0](https://crates.io/crates/x11cap) ([before](https://tools.taskcluster.net/task-inspector/#7wn8cjqXSOaZfpekKRY-yw)) ([after](https://tools.taskcluster.net/task-inspector/#bA6LwPreTMa8R_zYNt8Z3w))
  - [use](https://github.com/bryal/X11Cap/blob/d11b7170e6fa7c1ab370c69887b9ce71a542335d/src/lib.rs#L41) unnecessary.

  Requires nightly.

#### Non-root regressions, sorted by rank:

* [glium-0.8.0](https://crates.io/crates/glium) ([before](https://tools.taskcluster.net/task-inspector/#m5yEIEu-QEeM_2t4_11Opg)) ([after](https://tools.taskcluster.net/task-inspector/#Wztxoh9SQ-GqA4F3inaR9Q))
* [mio-0.4.1](https://crates.io/crates/mio) ([before](https://tools.taskcluster.net/task-inspector/#RtT-HmwbTYuG0djpAkVLvA)) ([after](https://tools.taskcluster.net/task-inspector/#Lx1d3ukPSGyRIwIDt_w0gw))
* [piston_window-0.11.0](https://crates.io/crates/piston_window) ([before](https://tools.taskcluster.net/task-inspector/#QE421inlRgShgoXKcUkEEA)) ([after](https://tools.taskcluster.net/task-inspector/#wIKQPW_7TjmrztHQ4Kk3hw))
* [piston2d-gfx_graphics-0.4.0](https://crates.io/crates/piston2d-gfx_graphics) ([before](https://tools.taskcluster.net/task-inspector/#hIUDm8m6QrCdOpSF30aPjQ)) ([after](https://tools.taskcluster.net/task-inspector/#HOw14MCoQxGj7GjYIy-Lng))
* [piston-gfx_texture-0.2.0](https://crates.io/crates/piston-gfx_texture) ([before](https://tools.taskcluster.net/task-inspector/#om-wlRW-Tm65MTlrpa8u7Q)) ([after](https://tools.taskcluster.net/task-inspector/#m9e9Vx58RA6KhCljujzzMQ))
* [piston2d-glium_graphics-0.3.0](https://crates.io/crates/piston2d-glium_graphics) ([before](https://tools.taskcluster.net/task-inspector/#vHeYcL2gRT2aIz9JeksAfw)) ([after](https://tools.taskcluster.net/task-inspector/#yEKBSm1BQ_C0O-4GKhQgUQ))
* [html5ever-0.2.0](https://crates.io/crates/html5ever) ([before](https://tools.taskcluster.net/task-inspector/#C0yCazihTWa4x2GxCUxasQ)) ([after](https://tools.taskcluster.net/task-inspector/#Vbl4HjqcQlq4-sJ2m1yBnQ))
* [caribon-0.6.2](https://crates.io/crates/caribon) ([before](https://tools.taskcluster.net/task-inspector/#AJZzG5gLSY-WVMKc-MoV5w)) ([after](https://tools.taskcluster.net/task-inspector/#ornLa3ZaSC-Zbz7ICg33Tg))
* [gj-0.0.2](https://crates.io/crates/gj) ([before](https://tools.taskcluster.net/task-inspector/#xhaiB76FQAKCEsmBkQtp1A)) ([after](https://tools.taskcluster.net/task-inspector/#rBJke3wpQqaq7wmEiQtLJA))
* [glium_text-0.5.0](https://crates.io/crates/glium_text) ([before](https://tools.taskcluster.net/task-inspector/#IMdXVtTYSIaDrCRQ6SbLTA)) ([after](https://tools.taskcluster.net/task-inspector/#t322h_mzQGarVmsf5MHqKA))
* [glyph_packer-0.0.0](https://crates.io/crates/glyph_packer) ([before](https://tools.taskcluster.net/task-inspector/#JmIVzau8RyOhnlTvdsRIHQ)) ([after](https://tools.taskcluster.net/task-inspector/#7k9GF09SQPya4ZrLuR6cJw))
* [html5ever_dom_sink-0.2.0](https://crates.io/crates/html5ever_dom_sink) ([before](https://tools.taskcluster.net/task-inspector/#7GJmaAYKS9WNqnbCx5XMrw)) ([after](https://tools.taskcluster.net/task-inspector/#pHotnKLkTAqK4-LP-n2MUQ))
* [identicon-0.1.0](https://crates.io/crates/identicon) ([before](https://tools.taskcluster.net/task-inspector/#15nnASVgStmrwqdCS1q8Rg)) ([after](https://tools.taskcluster.net/task-inspector/#WgJb_jEMQIebNgb_D2uq7Q))
* [assimp-0.0.4](https://crates.io/crates/assimp) ([before](https://tools.taskcluster.net/task-inspector/#-i-FYpJ2Rz-bcmxGVmxoOQ)) ([after](https://tools.taskcluster.net/task-inspector/#HXR8V8NeRMyOxF0Nnhdl0w))
* [jamkit-0.2.4](https://crates.io/crates/jamkit) ([before](https://tools.taskcluster.net/task-inspector/#mcpl8Z62Td-DFfoi9AqRnw)) ([after](https://tools.taskcluster.net/task-inspector/#XGOIXxqpRbCMy5bZ42GV5w))
* [coap-0.1.0](https://crates.io/crates/coap) ([before](https://tools.taskcluster.net/task-inspector/#SI137HlpRsSuQrlhxlRHpQ)) ([after](https://tools.taskcluster.net/task-inspector/#dT3pt46pQtmy3CvIaC_71Q))
* [kiss3d-0.1.2](https://crates.io/crates/kiss3d) ([before](https://tools.taskcluster.net/task-inspector/#2Bbro6uZQQCudv2ClalFTw)) ([after](https://tools.taskcluster.net/task-inspector/#9vRbugDKTDm94fjw6BcS6A))
  - [use](https://github.com/sebcrozet/kiss3d/blob/1c1d39d5f8a428609b2f7809c7237e8853ac24e9/src/text/glyph.rs#L7) seems to be unnecessary: semantically useless, just a space "optimisation", which actually makes no difference because the Vec field will be appropriately aligned always.

  On stable.
* [compass-sprite-0.0.3](https://crates.io/crates/compass-sprite) ([before](https://tools.taskcluster.net/task-inspector/#dTcfDsk1QYKWtK7EH5gnwg)) ([after](https://tools.taskcluster.net/task-inspector/#rElhdv9GS8-Zi14LSL-6Ng))
* [dcpu16-gui-0.0.3](https://crates.io/crates/dcpu16-gui) ([before](https://tools.taskcluster.net/task-inspector/#mtbOQfFUTDiZcMUc65LD3w)) ([after](https://tools.taskcluster.net/task-inspector/#co31ZVgNQ1mYyDCnSwBxJg))
* [piston3d-gfx_voxel-0.1.1](https://crates.io/crates/piston3d-gfx_voxel) ([before](https://tools.taskcluster.net/task-inspector/#2nZmq4zORIOdJ-ErCOCmww)) ([after](https://tools.taskcluster.net/task-inspector/#epzWs2zuSiWxfoWyMCv0Kw))
* [dev-0.0.7](https://crates.io/crates/dev) ([before](https://tools.taskcluster.net/task-inspector/#5hSafPV2RlKlubg7WHniPw)) ([after](https://tools.taskcluster.net/task-inspector/#ITQ6zXYpSAC3_AtmMe4xRw))
* [rustty-0.1.3](https://crates.io/crates/rustty) ([before](https://tools.taskcluster.net/task-inspector/#jlstxp6mSPqzQ1n3FgHSRA)) ([after](https://tools.taskcluster.net/task-inspector/#HgrQz6UVQ5yCkVX25Py-2w))
* [skeletal_animation-0.1.1](https://crates.io/crates/skeletal_animation) ([before](https://tools.taskcluster.net/task-inspector/#nyMUzqs6RZKIZJ1v1xcglA)) ([after](https://tools.taskcluster.net/task-inspector/#10lM9Vh5SBa7YD3swbm6pw))
* [slabmalloc-0.0.1](https://crates.io/crates/slabmalloc) ([before](https://tools.taskcluster.net/task-inspector/#li_vsJY8S9-OKEP_KIzEyQ)) ([after](https://tools.taskcluster.net/task-inspector/#1lcKVbKVQNqkKSfwEKIvkg))
* [spidev-0.1.0](https://crates.io/crates/spidev) ([before](https://tools.taskcluster.net/task-inspector/#5YidcvWyQ0KSmX_9yHjL5A)) ([after](https://tools.taskcluster.net/task-inspector/#mmDafSdlSIS-xfDvyeIckQ))
* [sysfs_gpio-0.3.2](https://crates.io/crates/sysfs_gpio) ([before](https://tools.taskcluster.net/task-inspector/#KEO87BJHSB-9wNHvTGgEiQ)) ([after](https://tools.taskcluster.net/task-inspector/#44Qnzq6CSBSrMti4utYEZQ))
* [texture_packer-0.0.1](https://crates.io/crates/texture_packer) ([before](https://tools.taskcluster.net/task-inspector/#-yNhXPaFSBK59eEPRBChVw)) ([after](https://tools.taskcluster.net/task-inspector/#dY5YnW-uTRuCAxxh93_P1w))
* [falcon-0.0.1](https://crates.io/crates/falcon) ([before](https://tools.taskcluster.net/task-inspector/#hsFGvgrWTL6yY5JVjm20Sw)) ([after](https://tools.taskcluster.net/task-inspector/#YMYfL2KkTH2fct8CD9nqUg))
* [filetype-0.2.0](https://crates.io/crates/filetype) ([before](https://tools.taskcluster.net/task-inspector/#bCC3ps_gT6m05BNm5lEnFw)) ([after](https://tools.taskcluster.net/task-inspector/#trGw9uPMTgiuxp-w821ZgA))

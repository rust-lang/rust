# Serialization in rustc

rustc has to [serialize] and deserialize various data during compilation.
Specifically:

- "Crate metadata", consisting mainly of query outputs, are serialized
  from a binary format into `rlib` and `rmeta` files that are output when
  compiling a library crate. These `rlib` and `rmeta` files are then
  deserialized by the crates which depend on that library.
- Certain query outputs are serialized in a binary format to
  [persist incremental compilation results].
- [`CrateInfo`] is serialized to `JSON` when the `-Z no-link` flag is used, and
  deserialized from `JSON` when the `-Z link-only` flag is used.

[`CrateInfo`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_codegen_ssa/struct.CrateInfo.html
[persist incremental compilation results]: queries/incremental-compilation-in-detail.md#the-real-world-how-persistence-makes-everything-complicated
[serialize]: https://en.wikipedia.org/wiki/Serialization

## The `Encodable` and `Decodable` traits

The [`rustc_serialize`] crate defines two traits for types which can be serialized:

```rust,ignore
pub trait Encodable<S: Encoder> {
    fn encode(&self, s: &mut S) -> Result<(), S::Error>;
}

pub trait Decodable<D: Decoder>: Sized {
    fn decode(d: &mut D) -> Result<Self, D::Error>;
}
```

It also defines implementations of these for various common standard library
[primitive types](https://doc.rust-lang.org/std/#primitives) such as integer
types, floating point types, `bool`, `char`, `str`, etc.

For types that are constructed from those types, `Encodable` and `Decodable`
are usually implemented by [derives]. These generate implementations that
forward deserialization to the fields of the struct or enum. For a
struct those impls look something like this:

```rust,ignore
#![feature(rustc_private)]
extern crate rustc_serialize;
use rustc_serialize::{Decodable, Decoder, Encodable, Encoder};

struct MyStruct {
    int: u32,
    float: f32,
}

impl<E: Encoder> Encodable<E> for MyStruct {
    fn encode(&self, s: &mut E) -> Result<(), E::Error> {
        s.emit_struct("MyStruct", 2, |s| {
            s.emit_struct_field("int", 0, |s| self.int.encode(s))?;
            s.emit_struct_field("float", 1, |s| self.float.encode(s))
        })
    }
}

impl<D: Decoder> Decodable<D> for MyStruct {
    fn decode(s: &mut D) -> Result<MyStruct, D::Error> {
        s.read_struct("MyStruct", 2, |d| {
            let int = d.read_struct_field("int", 0, Decodable::decode)?;
            let float = d.read_struct_field("float", 1, Decodable::decode)?;

            Ok(MyStruct { int, float })
        })
    }
}
```
[`rustc_serialize`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_serialize/index.html

## Encoding and Decoding arena allocated types

rustc has a lot of [arena allocated types].
Deserializing these types isn't possible without access to the arena that they need to be allocated on.
The [`TyDecoder`] and [`TyEncoder`] traits are supertraits of [`Decoder`] and [`Encoder`] that allow access to a [`TyCtxt`].

Types which contain `arena` allocated types can then bound the type parameter of their
[`Encodable`] and [`Decodable`] implementations with these traits.
For example

```rust,ignore
impl<'tcx, D: TyDecoder<'tcx>> Decodable<D> for MyStruct<'tcx> {
    /* ... */
}
```

The [`TyEncodable`] and [`TyDecodable`] [derive macros][derives] will expand to such
an implementation.

Decoding the actual `arena` allocated type is harder, because some of the
implementations can't be written due to the [orphan rules]. To work around this,
the [`RefDecodable`] trait is defined in [`rustc_middle`]. This can then be
implemented for any type. The `TyDecodable` macro will call `RefDecodable` to
decode references, but various generic code needs types to actually be
`Decodable` with a specific decoder.

For interned types instead of manually implementing `RefDecodable`, using a new
type wrapper, like [`ty::Predicate`] and manually implementing `Encodable` and
`Decodable` may be simpler.

[`Decodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_serialize/trait.Decodable.html
[`Decoder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_serialize/trait.Decoder.html
[`Encodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_serialize/trait.Encodable.html
[`Encoder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_serialize/trait.Encoder.html
[`RefDecodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/codec/trait.RefDecodable.html
[`rustc_middle`]: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_type_ir/codec.rs.html#21
[`ty::Predicate`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/predicate/struct.Predicate.html
[`TyCtxt`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/struct.TyCtxt.html
[`TyDecodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_macros/derive.TyDecodable.html
[`TyDecoder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/codec/trait.TyDecoder.html
[`TyEncodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_macros/derive.TyEncodable.html
[`TyEncoder`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_middle/ty/codec/trait.TyEncoder.html
[arena allocated types]: memory.md
[derives]: #derive-macros
[orphan rules]:https://doc.rust-lang.org/reference/items/implementations.html#orphan-rules

## Derive macros

The [`rustc_macros`] crate defines various derives to help implement `Decodable`
and `Encodable`.

- The `Encodable` and `Decodable` macros generate implementations that apply to
  all `Encoders` and `Decoders`. These should be used in crates that don't
  depend on [`rustc_middle`], or that have to be serialized by a type that does
  not implement `TyEncoder`.
- [`MetadataEncodable`] and [`MetadataDecodable`] generate implementations that
  only allow decoding by [`rustc_metadata::rmeta::encoder::EncodeContext`] and
  [`rustc_metadata::rmeta::decoder::DecodeContext`]. These are used for types
  that contain [`rustc_metadata::rmeta::`]`Lazy*`.
- `TyEncodable` and `TyDecodable` generate implementation that apply to any
  `TyEncoder` or `TyDecoder`. These should be used for types that are only
  serialized in crate metadata and/or the incremental cache, which is most
  serializable types in `rustc_middle`.

[`MetadataDecodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_macros/derive.MetadataDecodable.html
[`MetadataEncodable`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_macros/derive.MetadataEncodable.html
[`rustc_macros`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_macros
[`rustc_metadata::rmeta::`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/index.html
[`rustc_metadata::rmeta::decoder::DecodeContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/decoder/struct.DecodeContext.html
[`rustc_metadata::rmeta::encoder::EncodeContext`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/encoder/struct.EncodeContext.html
[`rustc_middle`]: https://github.com/rust-lang/rust/tree/master/compiler/rustc_middle

## Shorthands

`Ty` can be deeply recursive, if each `Ty` was encoded naively then crate
metadata would be very large. To handle this, each `TyEncoder` has a cache of
locations in its output where it has serialized types. If a type being encoded
is in the cache, then instead of serializing the type as usual, the byte offset
within the file being written is encoded instead. A similar scheme is used for
`ty::Predicate`.

## `LazyValue<T>`

Crate metadata is initially loaded before the `TyCtxt<'tcx>` is created, so
some deserialization needs to be deferred from the initial loading of metadata.
The [`LazyValue<T>`] type wraps the (relative) offset in the crate metadata
where a `T` has been serialized. There are also some variants, [`LazyArray<T>`]
and [`LazyTable<I, T>`].

The `LazyArray<[T]>` and `LazyTable<I, T>` types provide some functionality over
`Lazy<Vec<T>>` and `Lazy<HashMap<I, T>>`:

- It's possible to encode a `LazyArray<T>` directly from an `Iterator`, without
  first collecting into a `Vec<T>`.
- Indexing into a `LazyTable<I, T>` does not require decoding entries other
  than the one being read.

**note**: `LazyValue<T>` does not cache its value after being deserialized the
first time. Instead the query system itself is the main way of caching these
results.

[`LazyArray<T>`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/struct.LazyValue.html
[`LazyTable<I, T>`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/struct.LazyValue.html
[`LazyValue<T>`]: https://doc.rust-lang.org/nightly/nightly-rustc/rustc_metadata/rmeta/struct.LazyValue.html

## Specialization

A few types, most notably `DefId`, need to have different implementations for
different `Encoder`s. This is currently handled by ad-hoc specializations, for
example: `DefId` has a `default` implementation of `Encodable<E>` and a
specialized one for `Encodable<CacheEncoder>`.

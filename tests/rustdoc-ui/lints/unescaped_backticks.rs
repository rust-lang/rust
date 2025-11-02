#![deny(rustdoc::unescaped_backticks)]
#![allow(rustdoc::broken_intra_doc_links)]
#![allow(rustdoc::invalid_html_tags)]
#![allow(rustdoc::redundant_explicit_links)]

///
pub fn empty() {}

#[doc = ""]
pub fn empty2() {}

/// `
//~^ ERROR unescaped backtick
pub fn single() {}

/// \`
pub fn escaped() {}

/// \\`
//~^ ERROR unescaped backtick
pub fn not_escaped() {}

/// \\\`
pub fn not_not_escaped() {}

/// [`link1]
//~^ ERROR unescaped backtick
pub fn link1() {}

/// [link2`]
//~^ ERROR unescaped backtick
pub fn link2() {}

/// [`link_long](link_long)
//~^ ERROR unescaped backtick
pub fn link_long() {}

/// [`broken-link]
//~^ ERROR unescaped backtick
pub fn broken_link() {}

/// <xx:`>
pub fn url() {}

/// <x:`>
//~^ ERROR unescaped backtick
pub fn not_url() {}

/// <h1>`</h1>
pub fn html_tag() {}

/// &#96;
pub fn html_escape() {}

/// 🦀`🦀
//~^ ERROR unescaped backtick
pub fn unicode() {}

/// `foo(
//~^ ERROR unescaped backtick
///
/// paragraph
pub fn paragraph() {}

/// `foo `bar`
//~^ ERROR unescaped backtick
///
/// paragraph
pub fn paragraph2() {}

/// `foo(
//~^ ERROR unescaped backtick
/// not paragraph
pub fn not_paragraph() {}

/// Addition is commutative, which means that add(a, b)` is the same as `add(b, a)`.
//~^ ERROR unescaped backtick
///
/// You could use this function to add 42 to a number `n` (add(n, 42)`),
/// or even to add a number `n` to 42 (`add(42, b)`)!
//~^ ERROR unescaped backtick
pub fn add1(a: i32, b: i32) -> i32 { a + b }

/// Addition is commutative, which means that `add(a, b) is the same as `add(b, a)`.
//~^ ERROR unescaped backtick
///
/// You could use this function to add 42 to a number `n` (`add(n, 42)),
/// or even to add a number `n` to 42 (`add(42, n)`)!
//~^ ERROR unescaped backtick
pub fn add2(a: i32, b: i32) -> i32 { a + b }

/// Addition is commutative, which means that `add(a, b)` is the same as add(b, a)`.
//~^ ERROR unescaped backtick
///
/// You could use this function to add 42 to a number `n` (`add(n, 42)`),
/// or even to add a number `n` to 42 (add(42, n)`)!
//~^ ERROR unescaped backtick
pub fn add3(a: i32, b: i32) -> i32 { a + b }

/// Addition is commutative, which means that `add(a, b)` is the same as `add(b, a).
//~^ ERROR unescaped backtick
///
/// You could use this function to add 42 to a number `n` (`add(n, 42)),
/// or even to add a number `n` to 42 (`add(42, n)`)!
//~^ ERROR unescaped backtick
pub fn add4(a: i32, b: i32) -> i32 { a + b }

#[doc = "`"]
//~^ ERROR unescaped backtick
pub fn attr() {}

#[doc = concat!("\\", "`")]
pub fn attr_escaped() {}

#[doc = concat!("\\\\", "`")]
//~^ ERROR unescaped backtick
pub fn attr_not_escaped() {}

#[doc = "Addition is commutative, which means that add(a, b)` is the same as `add(b, a)`."]
//~^ ERROR unescaped backtick
pub fn attr_add1(a: i32, b: i32) -> i32 { a + b }

#[doc = "Addition is commutative, which means that `add(a, b) is the same as `add(b, a)`."]
//~^ ERROR unescaped backtick
pub fn attr_add2(a: i32, b: i32) -> i32 { a + b }

#[doc = "Addition is commutative, which means that `add(a, b)` is the same as add(b, a)`."]
//~^ ERROR unescaped backtick
pub fn attr_add3(a: i32, b: i32) -> i32 { a + b }

#[doc = "Addition is commutative, which means that `add(a, b)` is the same as `add(b, a)."]
//~^ ERROR unescaped backtick
pub fn attr_add4(a: i32, b: i32) -> i32 { a + b }

/// ``double backticks``
/// `foo
//~^ ERROR unescaped backtick
pub fn double_backticks() {}

/// # `(heading
//~^ ERROR unescaped backtick
/// ## heading2)`
//~^ ERROR unescaped backtick
///
/// multi `(
//~^ ERROR unescaped backtick
/// line
/// ) heading
/// =
///
/// para)`(graph
//~^ ERROR unescaped backtick
///
/// para)`(graph2
//~^ ERROR unescaped backtick
///
/// 1. foo)`
//~^ ERROR unescaped backtick
/// 2. `(bar
//~^ ERROR unescaped backtick
/// * baz)`
//~^ ERROR unescaped backtick
/// * `(quux
//~^ ERROR unescaped backtick
///
/// `#![this_is_actually_an_image(and(not), an = "attribute")]
//~^ ERROR unescaped backtick
///
/// #![this_is_actually_an_image(and(not), an = "attribute")]`
//~^ ERROR unescaped backtick
///
/// [this_is_actually_an_image(and(not), an = "attribute")]: `.png
///
/// | `table( | )head` |
//~^ ERROR unescaped backtick
//~| ERROR unescaped backtick
/// |---------|--------|
/// | table`( | )`body |
//~^ ERROR unescaped backtick
//~| ERROR unescaped backtick
pub fn complicated_markdown() {}

/// The `custom_mir` attribute tells the compiler to treat the function as being custom MIR. This
/// attribute only works on functions - there is no way to insert custom MIR into the middle of
/// another function. The `dialect` and `phase` parameters indicate which [version of MIR][dialect
/// docs] you are inserting here. Generally you'll want to use `#![custom_mir(dialect = "built")]`
/// if you want your MIR to be modified by the full MIR pipeline, or `#![custom_mir(dialect =
//~^ ERROR unescaped backtick
/// "runtime", phase = "optimized")] if you don't.
pub mod mir {}

pub mod rustc {
    /// Constructs a `TyKind::Error` type and registers a `span_delayed_bug` with the given `msg to
    //~^ ERROR unescaped backtick
    /// ensure it gets used.
    pub fn ty_error_with_message() {}

    pub struct WhereClause {
        /// `true` if we ate a `where` token: this can happen
        /// if we parsed no predicates (e.g. `struct Foo where {}
        /// This allows us to accurately pretty-print
        /// in `nt_to_tokenstream`
        //~^ ERROR unescaped backtick
        pub has_where_token: bool,
    }

    /// A symbol is an interned or gensymed string. The use of `newtype_index!` means
    /// that `Option<Symbol>` only takes up 4 bytes, because `newtype_index! reserves
    //~^ ERROR unescaped backtick
    /// the last 256 values for tagging purposes.
    pub struct Symbol();

    /// It is equivalent to `OpenOptions::new()` but allows you to write more
    /// readable code. Instead of `OpenOptions::new().read(true).open("foo.txt")`
    /// you can write `File::with_options().read(true).open("foo.txt"). This
    /// also avoids the need to import `OpenOptions`.
    //~^ ERROR unescaped backtick
    pub fn with_options() {}

    /// Subtracts `set from `row`. `set` can be either `BitSet` or
    /// `ChunkedBitSet`. Has no effect if `row` does not exist.
    //~^ ERROR unescaped backtick
    ///
    /// Returns true if the row was changed.
    pub fn subtract_row() {}

    pub mod assert_module_sources {
        //! The reason that we use `cfg=...` and not `#[cfg_attr]` is so that
        //! the HIR doesn't change as a result of the annotations, which might
        //! perturb the reuse results.
        //!
        //! `#![rustc_expected_cgu_reuse(module="spike", cfg="rpass2", kind="post-lto")]
        //~^ ERROR unescaped backtick
        //! allows for doing a more fine-grained check to see if pre- or post-lto data
        //! was re-used.

        /// `cfg=...
        //~^ ERROR unescaped backtick
        pub fn foo() {}

        /// `cfg=... and not `#[cfg_attr]`
        //~^ ERROR unescaped backtick
        pub fn bar() {}
    }

    /// Conceptually, this is like a `Vec<Vec<RWU>>`. But the number of
    /// RWU`s can get very large, so it uses a more compact representation.
    //~^ ERROR unescaped backtick
    pub struct RWUTable {}

    /// Like [Self::canonicalize_query], but preserves distinct universes. For
    /// example, canonicalizing `&'?0: Trait<'?1>`, where `'?0` is in `U1` and
    /// `'?1` is in `U3` would be canonicalized to have ?0` in `U1` and `'?1`
    /// in `U2`.
    //~^ ERROR unescaped backtick
    ///
    /// This is used for Chalk integration.
    pub fn canonicalize_query_preserving_universes() {}

    /// Note that we used to return `Error` here, but that was quite
    /// dubious -- the premise was that an error would *eventually* be
    /// reported, when the obligation was processed. But in general once
    /// you see an `Error` you are supposed to be able to assume that an
    /// error *has been* reported, so that you can take whatever heuristic
    /// paths you want to take. To make things worse, it was possible for
    /// cycles to arise, where you basically had a setup like `<MyType<$0>
    /// as Trait>::Foo == $0`. Here, normalizing `<MyType<$0> as
    /// Trait>::Foo> to `[type error]` would lead to an obligation of
    /// `<MyType<[type error]> as Trait>::Foo`. We are supposed to report
    /// an error for this obligation, but we legitimately should not,
    /// because it contains `[type error]`. Yuck! (See issue #29857 for
    //~^ ERROR unescaped backtick
    /// one case where this arose.)
    pub fn normalize_to_error() {}

    /// you don't want to cache that `B: AutoTrait` or `A: AutoTrait`
    /// is `EvaluatedToOk`; this is because they were only considered
    /// ok on the premise that if `A: AutoTrait` held, but we indeed
    /// encountered a problem (later on) with `A: AutoTrait. So we
    /// currently set a flag on the stack node for `B: AutoTrait` (as
    /// well as the second instance of `A: AutoTrait`) to suppress
    //~^ ERROR unescaped backtick
    /// caching.
    pub struct TraitObligationStack;

    /// Extend `scc` so that it can outlive some placeholder region
    /// from a universe it can't name; at present, the only way for
    /// this to be true is if `scc` outlives `'static`. This is
    /// actually stricter than necessary: ideally, we'd support bounds
    /// like `for<'a: 'b`>` that might then allow us to approximate
    /// `'a` with `'b` and not `'static`. But it will have to do for
    //~^ ERROR unescaped backtick
    /// now.
    pub fn add_incompatible_universe(){}
}

/// The Subscriber` may be accessed by calling [`WeakDispatch::upgrade`],
/// which returns an `Option<Dispatch>`. If all [`Dispatch`] clones that point
/// at the `Subscriber` have been dropped, [`WeakDispatch::upgrade`] will return
/// `None`. Otherwise, it will return `Some(Dispatch)`.
//~^ ERROR unescaped backtick
///
/// Returns some reference to this `[`Subscriber`] value if it is of type `T`,
/// or `None` if it isn't.
//~^ ERROR unescaped backtick
///
/// Called before the filtered [`Layer]'s [`on_event`], to determine if
/// `on_event` should be called.
//~^ ERROR unescaped backtick
///
/// Therefore, if the `Filter will change the value returned by this
/// method, it is responsible for ensuring that
/// [`rebuild_interest_cache`][rebuild] is called after the value of the max
//~^ ERROR unescaped backtick
/// level changes.
pub mod tracing {}

macro_rules! id {
    ($($tt:tt)*) => { $($tt)* }
}

id! {
    /// The Subscriber` may be accessed by calling [`WeakDispatch::upgrade`],
    //~^ ERROR unescaped backtick
    //~| ERROR unescaped backtick
    //~| ERROR unescaped backtick
    //~| ERROR unescaped backtick
    /// which returns an `Option<Dispatch>`. If all [`Dispatch`] clones that point
    /// at the `Subscriber` have been dropped, [`WeakDispatch::upgrade`] will return
    /// `None`. Otherwise, it will return `Some(Dispatch)`.
    ///
    /// Returns some reference to this `[`Subscriber`] value if it is of type `T`,
    /// or `None` if it isn't.
    ///
    /// Called before the filtered [`Layer]'s [`on_event`], to determine if
    /// `on_event` should be called.
    ///
    /// Therefore, if the `Filter will change the value returned by this
    /// method, it is responsible for ensuring that
    /// [`rebuild_interest_cache`][rebuild] is called after the value of the max
    /// level changes.
    pub mod tracing_macro {}
}

/// Regression test for <https://github.com/rust-lang/rust/issues/111117>
pub mod trillium_server_common {
    /// One-indexed, because the first CloneCounter is included. If you don't
    /// want the original to count, construct a [``CloneCounterObserver`]
    /// instead and use [`CloneCounterObserver::counter`] to increment.
    //~^ ERROR unescaped backtick
    pub struct CloneCounter;

    /// This is used by the above.
    pub struct CloneCounterObserver;
}

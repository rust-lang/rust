# Panicking in Rust

## Step 1: Invocation of the `panic!` macro.

There are actually two panic macros - one defined in `core`, and one defined in `std`.
This is due to the fact that code in `core` can panic. `core` is built before `std`,
but we want panics to use the same machinery at runtime, whether they originate in `core`
or `std`.

### core definition of panic!

The `core` `panic!` macro eventually makes the following call (in `library/core/src/panicking.rs`):

```rust
// NOTE This function never crosses the FFI boundary; it's a Rust-to-Rust call
extern "Rust" {
    #[lang = "panic_impl"]
    fn panic_impl(pi: &PanicInfo<'_>) -> !;
}

let pi = PanicInfo::internal_constructor(Some(&fmt), location);
unsafe { panic_impl(&pi) }
```

Actually resolving this goes through several layers of indirection:

1. In `compiler/rustc_middle/src/middle/weak_lang_items.rs`, `panic_impl` is
   declared as 'weak lang item', with the symbol `rust_begin_unwind`. This is
   used in `rustc_hir_analysis/src/collect.rs` to set the actual symbol name to
   `rust_begin_unwind`.

   Note that `panic_impl` is declared in an `extern "Rust"` block,
   which means that core will attempt to call a foreign symbol called `rust_begin_unwind`
   (to be resolved at link time)

2. In `library/std/src/panicking.rs`, we have this definition:

```rust
/// Entry point of panic from the core crate.
#[cfg(not(test))]
#[panic_handler]
#[unwind(allowed)]
pub fn begin_panic_handler(info: &PanicInfo<'_>) -> ! {
    ...
}
```

The special `panic_handler` attribute is resolved via `compiler/rustc_middle/src/middle/lang_items`.
The `extract` function converts the `panic_handler` attribute to a `panic_impl` lang item.

Now, we have a matching `panic_handler` lang item in the `std`. This function goes
through the same process as the `extern { fn panic_impl }` definition in `core`, ending
up with a symbol name of `rust_begin_unwind`. At link time, the symbol reference in `core`
will be resolved to the definition of `std` (the function called `begin_panic_handler` in the
Rust source).

Thus, control flow will pass from core to std at runtime. This allows panics from `core`
to go through the same infrastructure that other panics use (panic hooks, unwinding, etc)

### std implementation of panic!

This is where the actual panic-related logic begins. In `library/std/src/panicking.rs`,
control passes to `rust_panic_with_hook`. This method is responsible
for invoking the global panic hook, and checking for double panics. Finally,
we call `__rust_start_panic`, which is provided by the panic runtime.

The call to `__rust_start_panic` is very weird - it is passed a `*mut &mut dyn PanicPayload`,
converted to an `usize`. Let's break this type down:

1. `PanicPayload` is an internal trait. It is implemented for `PanicPayload`
(a wrapper around the user-supplied payload type), and has a method
`fn take_box(&mut self) -> *mut (dyn Any + Send)`.
This method takes the user-provided payload (`T: Any + Send`),
boxes it, and converts the box to a raw pointer.

2. When we call `__rust_start_panic`, we have an `&mut dyn PanicPayload`.
However, this is a fat pointer (twice the size of a `usize`).
To pass this to the panic runtime across an FFI boundary, we take a mutable
reference *to this mutable reference* (`&mut &mut dyn PanicPayload`), and convert it to a raw
pointer (`*mut &mut dyn PanicPayload`). The outer raw pointer is a thin pointer, since it points to
a `Sized` type (a mutable reference). Therefore, we can convert this thin pointer into a `usize`,
which is suitable for passing across an FFI boundary.

Finally, we call `__rust_start_panic` with this `usize`. We have now entered the panic runtime.

## Step 2: The panic runtime

Rust provides two panic runtimes: `panic_abort` and `panic_unwind`. The user chooses
between them at build time via their `Cargo.toml`

`panic_abort` is extremely simple: its implementation of `__rust_start_panic` just aborts,
as you would expect.

`panic_unwind` is the more interesting case.

In its implementation of `__rust_start_panic`, we take the `usize`, convert
it back to a `*mut &mut dyn PanicPayload`, dereference it, and call `take_box`
on the `&mut dyn PanicPayload`. At this point, we have a raw pointer to the payload
itself (a `*mut (dyn Send + Any)`): that is, a raw pointer to the actual value
provided by the user who called `panic!`.

At this point, the platform-independent code ends. We now call into
platform-specific unwinding logic (e.g `unwind`). This code is
responsible for unwinding the stack, running any 'landing pads' associated
with each frame (currently, running destructors), and transferring control
to the `catch_unwind` frame.

Note that all panics either abort the process or get caught by some call to `catch_unwind`.
In particular, in std's [runtime service],
the call to the user-provided `main` function is wrapped in `catch_unwind`.


[runtime service]: https://github.com/rust-lang/rust/blob/master/library/std/src/rt.rs

# `used`

The tracking issue for this feature
is: [40289](https://github.com/rust-lang/rust/issues/40289).

------------------------

The `#[used]` attribute can be applied to `static` variables to prevent the Rust
compiler from optimizing them away even if they appear to be unused by the crate
(appear to be "dead code").

``` rust
#![feature(used)]

#[used]
static FOO: i32 = 1;

static BAR: i32 = 2;

fn main() {}
```

If you compile this program into an object file, you'll see that `FOO` makes it
to the object file but `BAR` doesn't. Neither static variable is used by the
program.

``` text
$ rustc -C opt-level=3 --emit=obj used.rs

$ nm -C used.o
0000000000000000 T main
                 U std::rt::lang_start
0000000000000000 r used::FOO
0000000000000000 t used::main
```

Note that the *linker* knows nothing about the `#[used]` attribute and will
remove `#[used]` symbols if they are not referenced by other parts of the
program:

``` text
$ rustc -C opt-level=3 used.rs

$ nm -C used | grep FOO
```

"This doesn't sound too useful then!" you may think but keep reading.

To preserve the symbols all the way to the final binary, you'll need the
cooperation of the linker. Here's one example:

The ELF standard defines two special sections, `.init_array` and
`.pre_init_array`, that may contain function pointers which will be executed
*before* the `main` function is invoked. The linker will preserve symbols placed
in these sections (at least when linking programs that target the `*-*-linux-*`
targets).

``` rust,ignore
#![feature(used)]

extern "C" fn before_main() {
    println!("Hello, world!");
}

#[link_section = ".init_array"]
#[used]
static INIT_ARRAY: [extern "C" fn(); 1] = [before_main];

fn main() {}
```

So, `#[used]` and `#[link_section]` can be combined to obtain "life before
main".

``` text
$ rustc -C opt-level=3 before-main.rs

$ ./before-main
Hello, world!
```

Another example: ARM Cortex-M microcontrollers need their reset handler, a
pointer to the function that will executed right after the microcontroller is
turned on, to be placed near the start of their FLASH memory to boot properly.

This condition can be met using `#[used]` and `#[link_section]` plus a linker
script.

``` rust,ignore
#![feature(panic_implementation)]
#![feature(used)]
#![no_main]
#![no_std]

use core::panic::PanicInfo;

extern "C" fn reset_handler() -> ! {
    loop {}
}

#[link_section = ".reset_handler"]
#[used]
static RESET_HANDLER: extern "C" fn() -> ! = reset_handler;

#[panic_implementation]
fn panic_impl(info: &PanicInfo) -> ! {
    loop {}
}
```

``` text
MEMORY
{
  FLASH : ORIGIN = 0x08000000, LENGTH = 128K
  RAM : ORIGIN = 0x20000000, LENGTH = 20K
}

SECTIONS
{
  .text ORIGIN(FLASH) :
  {
    /* Vector table */
    LONG(ORIGIN(RAM) + LENGTH(RAM)); /* initial SP value */
    KEEP(*(.reset_handler));

    /* Omitted: The rest of the vector table */

    *(.text.*);
  } > FLASH

  /DISCARD/ :
  {
    /* Unused unwinding stuff */
    *(.ARM.exidx.*)
  }
}
```

``` text
$ xargo rustc --target thumbv7m-none-eabi --release -- \
    -C link-arg=-Tlink.x -C link-arg=-nostartfiles

$ arm-none-eabi-objdump -Cd target/thumbv7m-none-eabi/release/app
./target/thumbv7m-none-eabi/release/app:     file format elf32-littlearm


Disassembly of section .text:

08000000 <app::RESET_HANDLER-0x4>:
 8000000:       20005000        .word   0x20005000

08000004 <app::RESET_HANDLER>:
 8000004:       08000009                                ....

08000008 <app::reset_handler>:
 8000008:       e7fe            b.n     8000008 <app::reset_handler>
```

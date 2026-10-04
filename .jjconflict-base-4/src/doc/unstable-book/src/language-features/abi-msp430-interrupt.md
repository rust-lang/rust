# `abi_msp430_interrupt`

The tracking issue for this feature is: [#38487]

[#38487]: https://github.com/rust-lang/rust/issues/38487

------------------------

In the MSP430 architecture, interrupt handlers have a special calling
convention. You can use the `"msp430-interrupt"` ABI to make the compiler apply
the right calling convention to the interrupt handlers you define.

<!-- NOTE(ignore) this example is specific to the msp430 target -->

``` rust,ignore
#![feature(abi_msp430_interrupt)]
#![no_std]

// Place the interrupt handler at the appropriate memory address
// (Alternatively, you can use `#[used]` and remove `pub` and `#[no_mangle]`)
#[link_section = "__interrupt_vector_10"]
#[no_mangle]
pub static TIM0_VECTOR: extern "msp430-interrupt" fn() = tim0;

// The interrupt handler
extern "msp430-interrupt" fn tim0() {
    // ..
}
```

``` text
$ msp430-elf-objdump -CD ./target/msp430/release/app
Disassembly of section __interrupt_vector_10:

0000fff2 <TIM0_VECTOR>:
    fff2:       00 c0           interrupt service routine at 0xc000

Disassembly of section .text:

0000c000 <int::tim0>:
    c000:       00 13           reti
```

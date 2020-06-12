// Test that the AVR interrupt ABI cannot be used when avr_interrupt
// feature gate is not used.

extern "avr-interrupt" fn foo() {}
//~^ ERROR avr-interrupt and avr-non-blocking-interrupt ABIs are experimental and subject to change

fn main() {
    foo();
}

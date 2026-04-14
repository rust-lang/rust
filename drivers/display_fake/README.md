# display_fake

Loopback fake display driver for protocol negotiation and framing tests.

## Configuration
The driver reads the upper 32 bits of the `arg` word (passed at spawn) as a bitfield:

- bits 0-1: caps mask (bit0=DIRTY_RECTS, bit1=FULLFRAME; 0 => default both)
- bits 8-15: max_rects (0 => default 8)
- bit 16: split_writes (send WELCOME in 3 fragments)
- bit 17: burst (send ACKs back-to-back in one write)

## Behavior
- Expects HELLO, replies WELCOME (caps negotiated and max_rects applied)
- Accepts BIND, replies ACK
- Accepts PRESENT, replies ACK (ERR if unbound)

This driver is intended for deterministic, adversarial protocol testing.

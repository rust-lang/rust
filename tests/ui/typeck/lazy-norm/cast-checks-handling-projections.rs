// compile-flags: -Ztrait-solver=next
// known-bug: unknown

fn main() {
    (0u8 + 0u8) as char;
}

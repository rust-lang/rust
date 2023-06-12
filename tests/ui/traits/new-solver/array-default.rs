// compile-flags: -Ztrait-solver=next
// check-pass

fn has_default<const N: usize>() where [(); N]: Default {}

fn main() {
    has_default::<1>();
}

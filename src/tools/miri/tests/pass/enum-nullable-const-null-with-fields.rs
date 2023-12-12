static C: Result<(), Box<isize>> = Ok(());

// This is because of yet another bad assertion (ICE) about the null side of a nullable enum.
// So we won't actually compile if the bug is present, but we check the value in main anyway.

pub fn main() {
    assert!(C.is_ok());
}

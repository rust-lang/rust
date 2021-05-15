// rustfmt-version: Two

fn main() {
    assert!(
        HAYSTACK
            .par_iter()
            .find_any(|&&x| x[0] % 1000 == 999)
            .is_some()
    );

    assert(
        HAYSTACK
            .par_iter()
            .find_any(|&&x| x[0] % 1000 == 999)
            .is_some(),
    );
}

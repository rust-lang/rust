// check-pass
// known-bug

#![feature(let_chains)]

fn main() {
    let _opt = Some(1i32);

    #[cfg(FALSE)]
    {
        if let Some(elem) = _opt && {
            [1, 2, 3][let _ = ()];
            true
        } {
        }
    }
}

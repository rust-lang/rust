//! Regression test for <https://github.com/rust-lang/rust/issues/81097>.

fn main() {
    drop::<[(), 0]>([]);
    //~^ ERROR expected `;` or `]`, found `,`
}

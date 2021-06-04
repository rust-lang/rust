// rustfmt-use_field_init_shorthand: true
// Use field initialization shorthand if possible.

fn main() {
    let a = Foo { x, y, z };

    let b = Bar {
        x,
        y,
        #[attr]
        z,
        #[rustfmt::skip]
        skipped: skipped,
    };
}

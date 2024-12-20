//@ known-bug: #133426

fn a(
    _: impl Iterator<
        Item = [(); {
                   match *todo!() { ! };
               }],
    >,
) {
}

fn b(_: impl Iterator<Item = { match 0 { ! } }>) {}

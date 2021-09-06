// #47295: We used to have a hack of special-casing adjacent amtch
// arms whose patterns were composed solely of constants to not have
// them linked in the cfg.
//
// This was broken for various reasons. In particular, that hack was
// originally authored under the assunption that other checks
// elsewhere would ensure that the two patterns did not overlap.  But
// that assumption did not hold, at least not in the long run (namely,
// overlapping patterns were turned into warnings rather than errors).

#![feature(box_syntax)]

fn main() {
    let x: Box<_> = box 1;

    let v = (1, 2);

    match v {
        (1, 2) if take(x) => (),
        (1, 2) if take(x) => (), //~ ERROR use of moved value: `x`
        _ => (),
    }
}

fn take<T>(_: T) -> bool { false }

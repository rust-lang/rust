// unit-test: EarlyOtherwiseBranch

// FIXME: This test was broken by the derefer change.

// example from #68867
type CSSFloat = f32;

pub enum ViewportPercentageLength {
    Vw(CSSFloat),
    Vh(CSSFloat),
    Vmin(CSSFloat),
    Vmax(CSSFloat),
}

// EMIT_MIR early_otherwise_branch_68867.try_sum.EarlyOtherwiseBranch.diff
#[no_mangle]
pub extern "C" fn try_sum(
    x: &ViewportPercentageLength,
    other: &ViewportPercentageLength,
) -> Result<ViewportPercentageLength, ()> {
    use self::ViewportPercentageLength::*;
    Ok(match (x, other) {
        (&Vw(one), &Vw(other)) => Vw(one + other),
        (&Vh(one), &Vh(other)) => Vh(one + other),
        (&Vmin(one), &Vmin(other)) => Vmin(one + other),
        (&Vmax(one), &Vmax(other)) => Vmax(one + other),
        _ => return Err(()),
    })
}

fn main() {
    try_sum(&ViewportPercentageLength::Vw(1.0), &ViewportPercentageLength::Vw(2.0));
}

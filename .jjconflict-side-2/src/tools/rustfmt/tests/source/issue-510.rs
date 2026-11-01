impl ISizeAndMarginsComputer for AbsoluteNonReplaced {
fn solve_inline_size_constraints(&self,
block: &mut BlockFlow,
input: &ISizeConstraintInput)
-> ISizeConstraintSolution {
let (inline_start,inline_size,margin_inline_start,margin_inline_end) =
match (inline_startssssssxxxxxxsssssxxxxxxxxxssssssxxx,inline_startssssssxxxxxxsssssxxxxxxxxxssssssxxx) {
(MaybeAuto::Auto, MaybeAuto::Auto, MaybeAuto::Auto) => {
let margin_start = inline_start_margin.specified_or_zero();
let margin_end = inline_end_margin.specified_or_zero();
// Now it is the same situation as inline-start Specified and inline-end
// and inline-size Auto.
//
// Set inline-end to zero to calculate inline-size.
let inline_size = block.get_shrink_to_fit_inline_size(available_inline_size -
(margin_start + margin_end));
(Au(0), inline_size, margin_start, margin_end)
}
};

        let (inline_start, inline_size, margin_inline_start, margin_inline_end) =
            match (inline_start, inline_end, computed_inline_size) {
                (MaybeAuto::Auto, MaybeAuto::Auto, MaybeAuto::Auto) => {
                    let margin_start = inline_start_margin.specified_or_zero();
                    let margin_end = inline_end_margin.specified_or_zero();
                    // Now it is the same situation as inline-start Specified and inline-end
                    // and inline-size Auto.
                    //
                    // Set inline-end to zero to calculate inline-size.
                    let inline_size =
                        block.get_shrink_to_fit_inline_size(available_inline_size -
                                                            (margin_start + margin_end));
                    (Au(0), inline_size, margin_start, margin_end)
                }
            };
}
}

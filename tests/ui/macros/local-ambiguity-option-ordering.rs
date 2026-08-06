// Test the order in which meta-variable options causing ambiguity are presented in error messages.

#![crate_type = "lib"]

macro_rules! ambiguity_1 {
    ($($i:ident)* $j:ident) => {};
}

ambiguity_1!(error); //~ ERROR local ambiguity

macro_rules! ambiguity_2 {
    ( $( $( ; $i:ident )? $( $j:ident )? ; )* ) => {};
}

ambiguity_2!( ; error ); //~ ERROR local ambiguity
//
// Parse tree:
// - `$(...)*`:
// - Try one repetition
//   - `$(; $i)?`:
//   - Try entering
//     - Parse `;`
// ----> `$i` can parse `error`
//   - Try ignoring
//     - `$($j)?`:
//     - Try entering
//       - `$j` cannot parse `;`
//       - failure
//     - Try ignoring
//       - Parse `;`
//       - `$(...)*`:
//       - Try another repetition
//         - `$(; $i)?`:
//         - Try entering
//           - Cannot parse `;`
//           - failure
//         - Try ignoring
//           - `$($j)?`:
//           - Try entering
// ------------> `$j` can parse `error`
//           - Try ignoring
//             - Cannot parse `;`
//             - failure
//       - Try ignoring
//         - EOF vs `error`
//         - failure
// - Try no repetitions
//   - EOF vs `;`
//   - failure

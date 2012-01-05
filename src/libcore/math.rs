// Generic functions that have been defined for all numeric types
//
// (may very well go away again soon)

/*
Function: min

Returns the minimum of two values
*/
pure fn min<T: copy>(x: T, y: T) -> T { x < y ? x : y }

/*
Function: max

Returns the maximum of two values
*/
pure fn max<T: copy>(x: T, y: T) -> T { x < y ? y : x }


//@ revisions: too-high not-power-of-2
//
//@ [too-high] compile-flags: -Cmin-function-alignment=16384
//@ [not-power-of-2] compile-flags: -Cmin-function-alignment=3

//~? ERROR a number that is a power of 2 between 1 and 8192 was expected

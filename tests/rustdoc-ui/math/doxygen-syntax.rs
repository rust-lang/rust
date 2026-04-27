//@ run-rustfix
#![doc(syntax="+tex_math_dollars")]
#![feature(rustdoc_texmath)]
#![deny(rustdoc::invalid_math)]

//! The distance between \f$(x_1,y_1)\f$ and \f$(x_2,y_2)\f$ is
//~^ ERROR unknown
//~| HELP formulas
//~| ERROR unknown
//~| HELP formulas
//! \f$\sqrt{(x_2-x_1)^2+(y_2-y_1)^2}\f$.
//~^ ERROR unknown
//~| HELP formulas

#![feature(rustdoc_texmath)]
#![deny(invalid_doc_attributes)]

#[doc(syntax(enable(tex_math_dollars), enable()))]
// no conflict
pub struct A;

#[doc(syntax(enable(tex_math_dollars), enable(tex_math_dollars)))]
//~^ ERROR syntax
pub struct B;

#[doc(syntax(enable(tex_math_dollars)))]
pub mod c {
    #![doc(syntax(enable(tex_math_dollars)))]
    //~^ ERROR syntax
}

#[doc(syntax(enable(tex_math_dollars)))]
#[doc(syntax(enable(tex_math_dollars)))]
//~^ ERROR syntax
pub mod d {}

#[doc(syntax(enable(tex_math_dollars), disable()))]
// no conflict
pub struct DA;

#[doc(syntax(enable(tex_math_dollars), disable(tex_math_dollars)))]
//~^ ERROR syntax
pub struct DB;

#[doc(syntax(enable(tex_math_dollars)))]
pub mod dc {
    #![doc(syntax(disable(tex_math_dollars)))]
    //~^ ERROR syntax
}

#[doc(syntax(enable(tex_math_dollars)))]
#[doc(syntax(disable(tex_math_dollars)))]
//~^ ERROR syntax
pub mod dd {}

#[doc(syntax)]
//~^ ERROR syntax
pub struct MalformedNoArgs;

#[doc(syntax="+tex_math_dollars")]
//~^ ERROR syntax
pub struct MalformedOld;

#[doc(syntax(enable))]
//~^ ERROR syntax
pub struct MalformedNoArgsEnable;

#[doc(syntax(enable(tex_math_dollars())))]
//~^ ERROR syntax
pub struct MalformedArgsToSyntax;

#![deny(rustdoc::invalid_html_tags)]

/// <p/>
//~^ ERROR invalid self-closing HTML tag `p`
pub struct A;

/// <p style/>
//~^ ERROR invalid self-closing HTML tag `p`
pub struct B;

/// <p style=""/>
//~^ ERROR invalid self-closing HTML tag `p`
pub struct C;

/// <p style="x"/>
//~^ ERROR invalid self-closing HTML tag `p`
pub struct D;

/// <p style="x/></p>
//~^ ERROR unclosed quoted HTML attribute
pub struct E;

/// <p style='x/></p>
//~^ ERROR unclosed quoted HTML attribute
pub struct F;

/// <p style="x/"></p>
pub struct G;

/// <p style="x/"/>
//~^ ERROR invalid self-closing HTML tag `p`
pub struct H;

/// <p / >
//~^ ERROR invalid self-closing HTML tag `p`
pub struct I;

/// <br/>
pub struct J;

/// <a href=/></a>
pub struct K;

/// <a href=//></a>
pub struct L;

/// <a href="/"/>
//~^ ERROR invalid self-closing HTML tag `a`
pub struct M;

/// <a href=x />
//~^ ERROR invalid self-closing HTML tag `a`
pub struct N;

/// <a href= />
//~^ ERROR invalid self-closing HTML tag `a`
pub struct O;

/// <a href=x/></a>
pub struct P;

/// <svg><rect width=1 height=1 /></svg>
pub struct Q;

/// <svg><rect width=1 height=/></svg>
//~^ ERROR unclosed HTML tag `rect`
pub struct R;

/// <svg / q>
pub struct S;

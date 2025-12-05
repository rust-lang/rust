#![deny(rustdoc::invalid_html_tags)]

//! <p>💩<p>
//~^ ERROR unclosed HTML tag `p`
//~^^ ERROR unclosed HTML tag `p`

/// <img><input>
/// <script>
/// <img><input>
/// </script>
/// <unknown>
//~^ ERROR unclosed HTML tag `unknown`
/// < ok
/// <script>
//~^ ERROR unclosed HTML tag `script`
pub fn foo() {}

/// <h1>
///   <h2>
//~^ ERROR unclosed HTML tag `h2`
///     <h3>
//~^ ERROR unclosed HTML tag `h3`
/// </h1>
/// </hello>
//~^ ERROR unopened HTML tag `hello`
pub fn bar() {}

/// <div>
///    <br/> <p>
//~^ ERROR unclosed HTML tag `p`
/// </div>
pub fn a() {}

/// <div>
///   <p>
///      <div></div>
///   </p>
/// </div>
pub fn b() {}

/// <div style="hello">
//~^ ERROR unclosed HTML tag `div`
///   <h3>
//~^ ERROR unclosed HTML tag `h3`
/// <script
//~^ ERROR incomplete HTML tag `script`
pub fn c() {}

// Unclosed tags shouldn't warn if they are nested inside a <script> elem.
/// <script>
///   <h3><div>
/// </script>
/// <script>
///   <div>
///     <p>
///   </div>
/// </script>
pub fn d() {}

// Unclosed tags shouldn't warn if they are nested inside a <style> elem.
/// <style>
///   <h3><div>
/// </style>
/// <stYle>
///   <div>
///     <p>
///   </div>
/// </style>
pub fn e() {}

// Closing tags need to have ">" at the end, otherwise it's not a closing tag!
/// <div></div >
/// <div></div
//~^ ERROR unclosed HTML tag `div`
//~| ERROR incomplete HTML tag `div`
pub fn f() {}

/// <!---->
/// <!-- -->
/// <!-- <div> -->
/// <!-- <!-- -->
pub fn g() {}

/// <!--
/// -->
pub fn h() {}

/// <!--
//~^ ERROR Unclosed HTML comment
pub fn i() {}

/// hello
///
/// ```
/// uiapp.run(&env::args().collect::<Vec<_>>());
/// ```
pub fn j() {}

// Check that nested codeblocks are working as well
/// hello
///
/// ``````markdown
/// normal markdown
///
/// ```
/// uiapp.run(&env::args().collect::<Vec<_>>());
/// ```
///
// <Vec<_> shouldn't warn!
/// ``````
pub fn k() {}

/// Web Components style <dashed-tags>
//~^ ERROR unclosed HTML tag `dashed-tags`
/// Web Components style </unopened-tag>
//~^ ERROR unopened HTML tag `unopened-tag`
pub fn m() {}

/// backslashed \<a href="">
pub fn no_error_1() {}

/// backslashed \<<a href="">
//~^ ERROR unclosed HTML tag `a`
pub fn p() {}

/// <svg width="512" height="512" viewBox="0 0 512" fill="none" xmlns="http://www.w3.org/2000/svg">
///     <rect
///        width="256"
///        height="256"
///        fill="#5064C8"
///        stroke="black"
///     />
/// </svg>
pub fn no_error_2() {}

/// <div>
///     <img
///         src="https://example.com/ferris.png"
///         width="512"
///         height="512"
///     />
/// </div>
pub fn no_error_3() {}

/// > <div
/// > class="foo">
/// > </div>
pub fn no_error_4() {}

/// unfinished ALLOWED_UNCLOSED
///
/// note: CommonMark doesn't allow an html block to start with a multiline tag,
/// so we use `<br>` a bunch to force these to be parsed as html blocks.
///
/// <br>
/// <img
//~^ ERROR incomplete HTML tag `img`
pub fn q() {}

/// nested unfinished ALLOWED_UNCLOSED
/// <p><img</p>
//~^ ERROR incomplete HTML tag `img`
pub fn r() {}

/// > <br>
/// > <img
//~^ ERROR incomplete HTML tag `img`
/// > href="#broken"
pub fn s() {}

/// <br>
/// <br<br>
//~^ ERROR incomplete HTML tag `br`
pub fn t() {}

/// <br>
/// <br
//~^ ERROR incomplete HTML tag `br`
pub fn u() {}

/// <a href=">" alt="<">html5 allows this</a>
pub fn no_error_5() {}

/// <br>
/// <img title="
/// html5
/// allows
/// multiline
/// attr
/// values
/// these are just text, not tags:
/// </div>
/// <p/>
/// <div>
/// ">
pub fn no_error_6() {}

/// <br>
/// <a href="data:text/html,<!DOCTYPE>
/// <html>
/// <body><b>this is allowed for some reason</b></body>
/// </html>
/// ">what</a>
pub fn no_error_7() {}

/// Technically this is allowed per the html5 spec,
/// but there's basically no legitemate reason to do it,
/// so we don't allow it.
///
/// <p <!-->foobar</p>
//~^ ERROR Unclosed HTML comment
//~| ERROR incomplete HTML tag `p`
pub fn v() {}

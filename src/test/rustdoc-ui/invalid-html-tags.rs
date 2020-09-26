#![deny(invalid_html_tags)]

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
pub fn f() {}

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
pub fn c() {}

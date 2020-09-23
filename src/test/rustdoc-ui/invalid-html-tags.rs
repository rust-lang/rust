#![deny(invalid_html_tags)]

/// <script>
//~^ ERROR unclosed HTML tag `unknown`
//~^^ ERROR unclosed HTML tag `script`
/// <img><input>
/// </script>
/// <unknown>
/// < ok
/// <script>
pub fn foo() {}

/// <h1>
//~^ ERROR unopened HTML tag `h2`
//~^^ ERROR unopened HTML tag `h3`
///   <h2>
///     <h3>
/// </h1>
pub fn f() {}

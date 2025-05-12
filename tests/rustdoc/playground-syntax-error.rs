#![crate_name = "foo"]
#![doc(html_playground_url = "https://play.rust-lang.org/")]

/// bar docs
///
/// ```edition2015
/// use std::future::Future;
/// use std::pin::Pin;
/// fn foo_recursive(n: usize) -> Pin<Box<dyn Future<Output = ()>>> {
///     Box::pin(async move {
///         if n > 0 {
///             foo_recursive(n - 1).await;
///         }
///     })
/// }
/// ```
pub fn bar() {}

//@ has foo/fn.bar.html
//@ has - '//a[@class="test-arrow"]' ""
//@ has - '//*[@class="docblock"]' 'foo_recursive'

// rustfmt-wrap_comments: true
// rustfmt-max_width: 89

// Code block in doc comments that will exceed max width.
/// ```rust
/// extern crate actix_web;
/// use actix_web::{actix, server, App, HttpResponse};
///
/// fn main() {
///     // Run actix system, this method actually starts all async processes
///     actix::System::run(|| {
///         server::new(|| App::new().resource("/", |r| r.h(|_| HttpResponse::Ok())))
///             .bind("127.0.0.1:0")
///             .expect("Can not bind to 127.0.0.1:0")
///             .start();
/// #           actix::Arbiter::system().do_send(actix::msgs::SystemExit(0));
///     });
/// }
/// ```
fn foo() {}

// Code block in doc comments without the closing '```'.
/// ```rust
/// # extern crate actix_web;
/// use actix_web::{App, HttpResponse, http};
///
/// fn main() {
///     let app = App::new()
///         .resource(
///             "/", |r| r.method(http::Method::GET).f(|r| HttpResponse::Ok()))
///         .finish();
/// }
fn bar() {}

// `#` with indent.
/// ```rust
/// # use std::thread;
/// # extern crate actix_web;
/// use actix_web::{server, App, HttpResponse};
///
/// struct State1;
///
/// struct State2;
///
/// fn main() {
///     # thread::spawn(|| {
///     server::new(|| {
///         vec![
///             App::with_state(State1)
///                 .prefix("/app1")
///                 .resource("/", |r| r.f(|r| HttpResponse::Ok()))
///                 .boxed(),
///             App::with_state(State2)
///                 .prefix("/app2")
///                 .resource("/", |r| r.f(|r| HttpResponse::Ok()))
///                 .boxed(),
///         ]
///     })
///     .bind("127.0.0.1:8080")
///     .unwrap()
///     .run()
///     # });
/// }
/// ```
fn foobar() {}

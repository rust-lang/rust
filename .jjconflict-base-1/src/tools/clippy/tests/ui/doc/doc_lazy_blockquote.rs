#![warn(clippy::doc_lazy_continuation)]

/// > blockquote with
/// lazy continuation
//~^ ERROR: doc quote line without `>` marker
fn first() {}

/// > blockquote with no
/// > lazy continuation
fn first_nowarn() {}

/// > blockquote with no
///
/// lazy continuation
fn two_nowarn() {}

/// > nest here
/// >
/// > > nest here
/// > lazy continuation
//~^ ERROR: doc quote line without `>` marker
fn two() {}

/// > nest here
/// >
/// > > nest here
/// lazy continuation
//~^ ERROR: doc quote line without `>` marker
fn three() {}

/// >   * > nest here
/// lazy continuation
//~^ ERROR: doc quote line without `>` marker
fn four() {}

/// > * > nest here
/// lazy continuation
//~^ ERROR: doc quote line without `>` marker
fn four_point_1() {}

/// * > nest here lazy continuation
fn five() {}

/// 1. > nest here
///  lazy continuation (this results in strange indentation, but still works)
//~^ ERROR: doc quote line without `>` marker
fn six() {}

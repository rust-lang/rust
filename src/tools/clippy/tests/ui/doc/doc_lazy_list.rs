#![warn(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

/// 1. nest here
/// lazy continuation
//~^ ERROR: doc list item without indentation
fn one() {}

/// 1. first line
/// lazy list continuations don't make warnings with this lint
//~^ ERROR: doc list item without indentation
/// because they don't have the
//~^ ERROR: doc list item without indentation
fn two() {}

///   - nest here
/// lazy continuation
//~^ ERROR: doc list item without indentation
fn three() {}

///   - first line
/// lazy list continuations don't make warnings with this lint
//~^ ERROR: doc list item without indentation
/// because they don't have the
//~^ ERROR: doc list item without indentation
fn four() {}

///   - nest here
/// lazy continuation
//~^ ERROR: doc list item without indentation
fn five() {}

///   - - first line
/// this will warn on the lazy continuation
//~^ ERROR: doc list item without indentation
///     and so should this
//~^ ERROR: doc list item without indentation
fn six() {}

///   - - first line
///
///     this is not a lazy continuation
fn seven() {}

#[rustfmt::skip]
// https://github.com/rust-lang/rust-clippy/pull/12770#issuecomment-2118601768
/// Returns a list of ProtocolDescriptors from a Serde JSON input.
///
/// Defined Protocol Identifiers for the Protocol Descriptor
/// We intentionally omit deprecated profile identifiers.
/// From Bluetooth Assigned Numbers:
/// https://www.bluetooth.com/specifications/assigned-numbers/service-discovery
///
/// # Arguments
/// * `protocol_descriptors`: A Json Representation of the ProtocolDescriptors
///     to set up. Example:
///  'protocol_descriptors': [
//~^ ERROR: doc list item without indentation
///      {
///          'protocol': 25,  # u64 Representation of ProtocolIdentifier::AVDTP
///          'params': [
///              {
///                 'data': 0x0103  # to indicate 1.3
///              },
///              {
///                  'data': 0x0105  # to indicate 1.5
///              }
///          ]
///      },
///      {
///          'protocol': 1,  # u64 Representation of ProtocolIdentifier::SDP
///          'params': [{
///              'data': 0x0019
///          }]
///      }
///  ]
//~^ ERROR: doc list item without indentation
fn eight() {}

#[rustfmt::skip]
// https://github.com/rust-lang/rust-clippy/issues/13705
/// - \[text in square brackets\] with a long following description
///   that goes over multiple lines
pub fn backslash_escaped_square_braces() {}

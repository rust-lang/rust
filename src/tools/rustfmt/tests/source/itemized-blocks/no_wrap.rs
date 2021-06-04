// rustfmt-normalize_comments: true
// rustfmt-format_code_in_doc_comments: true

//! This is a list:
//!  * Outer
//!  * Outer
//!   * Inner
//!   * Inner with lots of text so that it could be reformatted something something something lots of text so that it could be reformatted something something something
//!
//! This example shows how to configure fern to output really nicely colored logs
//! - when the log level is error, the whole line is red
//! - when the log level is warn, the whole line is yellow
//! - when the log level is info, the level name is green and the rest of the line is white
//! - when the log level is debug, the whole line is white
//! - when the log level is trace, the whole line is gray ("bright black")

/// All the parameters ***except for `from_theater`*** should be inserted as sent by the remote
/// theater, i.e., as passed to [`Theater::send`] on the remote actor:
///  * `from` is the sending (remote) [`ActorId`], as reported by the remote theater by theater-specific means
///  * `to` is the receiving (local) [`ActorId`], as requested by the remote theater
///  * `tag` is a tag that identifies the message type
///  * `msg` is the (serialized) message
/// All the parameters ***except for `from_theater`*** should be inserted as sent by the remote
/// theater, i.e., as passed to [`Theater::send`] on the remote actor
fn func1() {}

/// All the parameters ***except for `from_theater`*** should be inserted as sent by the remote
/// theater, i.e., as passed to [`Theater::send`] on the remote actor:
///  * `from` is the sending (remote) [`ActorId`], as reported by the remote theater by theater-specific means
///  * `to` is the receiving (local) [`ActorId`], as requested by the remote theater
///  * `tag` is a tag that identifies the message type
///  * `msg` is the (serialized) message
/// ```
/// let x =     42;
/// ```
fn func2() {}

/// Look:
///
/// ```
/// let x =     42;
/// ```
///  * `from` is the sending (remote) [`ActorId`], as reported by the remote theater by theater-specific means
///  * `to` is the receiving (local) [`ActorId`], as requested by the remote theater
///  * `tag` is a tag that identifies the message type
///  * `msg` is the (serialized) message
fn func3() {}

// rustfmt-wrap_comments: true
// rustfmt-format_code_in_doc_comments: true
// rustfmt-max_width: 50

//! This is an itemized markdown list (see also issue #3224):
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
//!
//! This is a numbered markdown list (see also issue #5416):
//! 1. Long long long long long long long long long long long long long long long long long line
//! 2. Another very long long long long long long long long long long long long long long long line
//! 3. Nested list
//!    1. Long long long long long long long long long long long long long long long long line
//!    2. Another very long long long long long long long long long long long long long long line
//! 4. Last item
//!
//! Using the ')' instead of '.' character after the number:
//! 1) Long long long long long long long long long long long long long long long long long line
//! 2) Another very long long long long long long long long long long long long long long long line
//!
//! Deep list that mixes various bullet and number formats:
//! 1. First level with a long long long long long long long long long long long long long long
//!    long long long line
//! 2. First level with another very long long long long long long long long long long long long
//!    long long long line
//!     * Second level with a long long long long long long long long long long long long long
//!       long long long line
//!     * Second level with another very long long long long long long long long long long long
//!       long long long line
//!         1) Third level with a long long long long long long long long long long long long long
//!            long long long line
//!         2) Third level with another very long long long long long long long long long long
//!            long long long long line
//!             - Forth level with a long long long long long long long long long long long long
//!               long long long long line
//!             - Forth level with another very long long long long long long long long long long
//!               long long long long line
//!         3) One more item at the third level
//!         4) Last item of the third level
//!     * Last item of second level
//! 3. Last item of first level

// This example shows how to configure fern to output really nicely colored logs
// - when the log level is error, the whole line is red
//   - when the log level is warn, the whole line is yellow
//     - when the log level is info, the level name is green and the rest of the line is white
//   - when the log level is debug, the whole line is white
//   - when the log level is trace, the whole line is gray ("bright black")

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

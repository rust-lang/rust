use super::Error;
use crate::fmt;

#[derive(Debug, PartialEq)]
struct A;
#[derive(Debug, PartialEq)]
struct B;

impl fmt::Display for A {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "A")
    }
}
impl fmt::Display for B {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "B")
    }
}

impl Error for A {}
impl Error for B {}

#[test]
fn downcasting() {
    let mut a = A;
    let a = &mut a as &mut (dyn Error + 'static);
    assert_eq!(a.downcast_ref::<A>(), Some(&A));
    assert_eq!(a.downcast_ref::<B>(), None);
    assert_eq!(a.downcast_mut::<A>(), Some(&mut A));
    assert_eq!(a.downcast_mut::<B>(), None);

    let a: Box<dyn Error> = Box::new(A);
    match a.downcast::<B>() {
        Ok(..) => panic!("expected error"),
        Err(e) => assert_eq!(*e.downcast::<A>().unwrap(), A),
    }
}

use crate::backtrace::Backtrace;
use crate::error::Report;

#[derive(Debug)]
struct SuperError {
    source: SuperErrorSideKick,
}

impl fmt::Display for SuperError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SuperError is here!")
    }
}

impl Error for SuperError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(&self.source)
    }
}

#[derive(Debug)]
struct SuperErrorSideKick;

impl fmt::Display for SuperErrorSideKick {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SuperErrorSideKick is here!")
    }
}

impl Error for SuperErrorSideKick {}

#[test]
fn single_line_formatting() {
    let error = SuperError { source: SuperErrorSideKick };
    let report = Report::new(&error);
    let actual = report.to_string();
    let expected = String::from("SuperError is here!: SuperErrorSideKick is here!");

    assert_eq!(expected, actual);
}

#[test]
fn multi_line_formatting() {
    let error = SuperError { source: SuperErrorSideKick };
    let report = Report::new(&error).pretty(true);
    let actual = report.to_string();
    let expected = String::from(
        "\
SuperError is here!

Caused by:
      SuperErrorSideKick is here!",
    );

    assert_eq!(expected, actual);
}

#[test]
fn error_with_no_sources_formats_single_line_correctly() {
    let report = Report::new(SuperErrorSideKick);
    let actual = report.to_string();
    let expected = String::from("SuperErrorSideKick is here!");

    assert_eq!(expected, actual);
}

#[test]
fn error_with_no_sources_formats_multi_line_correctly() {
    let report = Report::new(SuperErrorSideKick).pretty(true);
    let actual = report.to_string();
    let expected = String::from("SuperErrorSideKick is here!");

    assert_eq!(expected, actual);
}

#[test]
fn error_with_backtrace_outputs_correctly_with_one_source() {
    let trace = Backtrace::force_capture();
    let expected = format!(
        "\
The source of the error

Caused by:
      Error with backtrace

Stack backtrace:
{}",
        trace
    );
    let error = GenericError::new("Error with backtrace");
    let mut error = GenericError::new_with_source("The source of the error", error);
    error.backtrace = Some(trace);
    let report = Report::new(error).pretty(true).show_backtrace(true);

    println!("Error: {}", report);
    assert_eq!(expected.trim_end(), report.to_string());
}

#[test]
fn error_with_backtrace_outputs_correctly_with_two_sources() {
    let trace = Backtrace::force_capture();
    let expected = format!(
        "\
Error with two sources

Caused by:
   0: The source of the error
   1: Error with backtrace

Stack backtrace:
{}",
        trace
    );
    let mut error = GenericError::new("Error with backtrace");
    error.backtrace = Some(trace);
    let error = GenericError::new_with_source("The source of the error", error);
    let error = GenericError::new_with_source("Error with two sources", error);
    let report = Report::new(error).pretty(true).show_backtrace(true);

    println!("Error: {}", report);
    assert_eq!(expected.trim_end(), report.to_string());
}

#[derive(Debug)]
struct GenericError<D> {
    message: D,
    backtrace: Option<Backtrace>,
    source: Option<Box<dyn Error + 'static>>,
}

impl<D> GenericError<D> {
    fn new(message: D) -> GenericError<D> {
        Self { message, backtrace: None, source: None }
    }

    fn new_with_source<E>(message: D, source: E) -> GenericError<D>
    where
        E: Error + 'static,
    {
        let source: Box<dyn Error + 'static> = Box::new(source);
        let source = Some(source);
        GenericError { message, backtrace: None, source }
    }
}

impl<D> fmt::Display for GenericError<D>
where
    D: fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.message, f)
    }
}

impl<D> Error for GenericError<D>
where
    D: fmt::Debug + fmt::Display,
{
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.source.as_deref()
    }

    fn backtrace(&self) -> Option<&Backtrace> {
        self.backtrace.as_ref()
    }
}

#[test]
fn error_formats_single_line_with_rude_display_impl() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("line 1\nline 2")?;
            f.write_str("\nline 3\nline 4\n")?;
            f.write_str("line 5\nline 6")?;
            Ok(())
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error);
    let expected = "\
line 1
line 2
line 3
line 4
line 5
line 6: line 1
line 2
line 3
line 4
line 5
line 6: line 1
line 2
line 3
line 4
line 5
line 6: line 1
line 2
line 3
line 4
line 5
line 6";

    let actual = report.to_string();
    assert_eq!(expected, actual);
}

#[test]
fn error_formats_multi_line_with_rude_display_impl() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("line 1\nline 2")?;
            f.write_str("\nline 3\nline 4\n")?;
            f.write_str("line 5\nline 6")?;
            Ok(())
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error).pretty(true);
    let expected = "line 1
line 2
line 3
line 4
line 5
line 6

Caused by:
   0: line 1
      line 2
      line 3
      line 4
      line 5
      line 6
   1: line 1
      line 2
      line 3
      line 4
      line 5
      line 6
   2: line 1
      line 2
      line 3
      line 4
      line 5
      line 6";

    let actual = report.to_string();
    assert_eq!(expected, actual);
}

#[test]
fn errors_that_start_with_newline_formats_correctly() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("\nThe message\n")
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error).pretty(true);
    let expected = "
The message


Caused by:
   0: \
\n      The message
      \
\n   1: \
\n      The message
      ";

    let actual = report.to_string();
    assert_eq!(expected, actual);
}

#[test]
fn errors_with_multiple_writes_on_same_line_dont_insert_erroneous_newlines() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("The message")?;
            f.write_str(" goes on")?;
            f.write_str(" and on.")
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error).pretty(true);
    let expected = "\
The message goes on and on.

Caused by:
   0: The message goes on and on.
   1: The message goes on and on.";

    let actual = report.to_string();
    println!("{}", actual);
    assert_eq!(expected, actual);
}

#[test]
fn errors_with_string_interpolation_formats_correctly() {
    #[derive(Debug)]
    struct MyMessage(usize);

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "Got an error code: ({}). ", self.0)?;
            write!(f, "What would you like to do in response?")
        }
    }

    let error = GenericError::new(MyMessage(10));
    let error = GenericError::new_with_source(MyMessage(20), error);
    let report = Report::new(error).pretty(true);
    let expected = "\
Got an error code: (20). What would you like to do in response?

Caused by:
      Got an error code: (10). What would you like to do in response?";
    let actual = report.to_string();
    assert_eq!(expected, actual);
}

#[test]
fn empty_lines_mid_message() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("line 1\n\nline 2")
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error).pretty(true);
    let expected = "\
line 1

line 2

Caused by:
   0: line 1
      \
\n      line 2
   1: line 1
      \
\n      line 2";

    let actual = report.to_string();
    assert_eq!(expected, actual);
}

#[test]
fn only_one_source() {
    #[derive(Debug)]
    struct MyMessage;

    impl fmt::Display for MyMessage {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.write_str("line 1\nline 2")
        }
    }

    let error = GenericError::new(MyMessage);
    let error = GenericError::new_with_source(MyMessage, error);
    let report = Report::new(error).pretty(true);
    let expected = "\
line 1
line 2

Caused by:
      line 1
      line 2";

    let actual = report.to_string();
    assert_eq!(expected, actual);
}

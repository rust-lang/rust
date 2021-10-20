use core::panic::assert_info::{AssertInfo, Assertion, BinaryAssertion};
use core::panic::Location;

const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// Print an assertion to standard error.
///
/// If `ansi_colors` is true, this function unconditionally prints ANSI color codes.
/// It should only be set to true only if it is known that the terminal supports it.
pub fn pretty_print_assertion(assert: &AssertInfo<'_>, loc: Location<'_>, ansi_colors: bool) {
    if ansi_colors {
        print_pretty_header(loc);
        match assert.assertion {
            Assertion::Binary(ref assertion) => print_pretty_binary_assertion(assertion),
        };
        if let Some(msg) = &assert.message {
            print_pretty_message(msg);
        }
    } else {
        print_plain_header(loc);
        match assert.assertion {
            Assertion::Binary(ref assertion) => print_plain_binary_assertion(assertion),
        };
        if let Some(msg) = &assert.message {
            print_plain_message(msg);
        }
    }
}

fn print_plain_header(loc: Location<'_>) {
    eprintln!("Assertion failed at {}:{}:{}:", loc.file(), loc.line(), loc.column())
}

fn print_pretty_header(loc: Location<'_>) {
    eprintln!(
        "{bold}{red}Assertion failed{reset} at {bold}{file}{reset}:{line}:{col}:",
        red = RED,
        bold = BOLD,
        reset = RESET,
        file = loc.file(),
        line = loc.line(),
        col = loc.column(),
    );
}

fn print_plain_binary_assertion(assertion: &BinaryAssertion<'_>) {
    eprint!(
        concat!(
            "Assertion:\n",
            "  {macro_name}!( {left_expr}, {right_expr} )\n",
            "Expansion:\n",
            "  {macro_name}!( {left_val:?}, {right_val:?} )\n",
        ),
        macro_name = assertion.static_data.kind.macro_name(),
        left_expr = assertion.static_data.left_expr,
        right_expr = assertion.static_data.right_expr,
        left_val = assertion.left_val,
        right_val = assertion.right_val,
    );
}

fn print_pretty_binary_assertion(assertion: &BinaryAssertion<'_>) {
    eprint!(
        concat!(
            "{bold}Assertion:{reset}\n",
            "  {magenta}{macro_name}!( {cyan}{left_expr}{magenta}, {yellow}{right_expr} {magenta}){reset}\n",
            "{bold}Expansion:{reset}\n",
            "  {magenta}{macro_name}!( {cyan}{left_val:?}{magenta}, {yellow}{right_val:?} {magenta}){reset}\n",
        ),
        cyan = CYAN,
        magenta = MAGENTA,
        yellow = YELLOW,
        bold = BOLD,
        reset = RESET,
        macro_name = assertion.static_data.kind.macro_name(),
        left_expr = assertion.static_data.left_expr,
        right_expr = assertion.static_data.right_expr,
        left_val = assertion.left_val,
        right_val = assertion.right_val,
    );
}

fn print_plain_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("Message:\n  {}", message);
}

fn print_pretty_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("{bold}Message:{reset}\n  {msg}", bold = BOLD, reset = RESET, msg = message,);
}

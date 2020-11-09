use core::panic::assert_info::{AssertInfo, Assertion, BinaryAssertion, BoolAssertion};
use core::panic::Location;

const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
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
        match &assert.assertion {
            Assertion::Bool(assert) => print_pretty_bool_assertion(assert),
            Assertion::Binary(assert) => print_pretty_binary_assertion(assert),
        }
        if let Some(msg) = &assert.message {
            print_pretty_message(msg);
        }
    } else {
        print_plain_header(loc);
        match &assert.assertion {
            Assertion::Bool(assert) => print_plain_bool_assertion(assert),
            Assertion::Binary(assert) => print_plain_binary_assertion(assert),
        }
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

fn print_plain_bool_assertion(assert: &BoolAssertion) {
    eprintln!("Assertion:\n  {}", assert.expr);
    eprintln!("Expansion:\n  false");
}

fn print_pretty_bool_assertion(assert: &BoolAssertion) {
    eprintln!(
        "{bold}Assertion:{reset}\n  {cyan}{expr}{reset}",
        cyan = CYAN,
        reset = RESET,
        bold = BOLD,
        expr = assert.expr,
    );
    eprintln!(
        "{bold}Expansion:{reset}\n  {cyan}false{reset}",
        cyan = CYAN,
        bold = BOLD,
        reset = RESET,
    );
}

fn print_plain_binary_assertion(assert: &BinaryAssertion<'_>) {
    eprintln!("Assertion:\n  {} {} {}", assert.left_expr, assert.op, assert.right_expr);
    eprintln!("Expansion:\n  {:?} {} {:?}", assert.left_val, assert.op, assert.right_val);
}

fn print_pretty_binary_assertion(assert: &BinaryAssertion<'_>) {
    eprintln!(
        "{bold}Assertion:{reset}\n  {cyan}{left} {bold}{blue}{op}{reset} {yellow}{right}{reset}",
        cyan = CYAN,
        blue = BLUE,
        yellow = YELLOW,
        bold = BOLD,
        reset = RESET,
        left = assert.left_expr,
        op = assert.op,
        right = assert.right_expr,
    );
    eprintln!(
        "{bold}Expansion:{reset}\n  {cyan}{left:?} {bold}{blue}{op}{reset} {yellow}{right:?}{reset}",
        cyan = CYAN,
        blue = BLUE,
        yellow = YELLOW,
        bold = BOLD,
        reset = RESET,
        left = assert.left_val,
        op = assert.op,
        right = assert.right_val,
    );
}

fn print_plain_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("Message:\n  {}", message);
}

fn print_pretty_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("{bold}Message:{reset}\n  {msg}", bold = BOLD, reset = RESET, msg = message,);
}

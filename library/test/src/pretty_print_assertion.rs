use core::panic::assert_info::{AssertInfo, Assertion, BinaryAssertion, BoolAssertion};
use core::panic::Location;

const RED: &str = "\x1b[31m";
const YELLOW: &str = "\x1b[33m";
const BLUE: &str = "\x1b[34m";
const MAGENTA: &str = "\x1b[35m";
const CYAN: &str = "\x1b[36m";
const BOLD: &str = "\x1b[1m";
const RESET: &str = "\x1b[0m";

/// Print an assertion to standard error.
///
/// If `ansi_colors` is true, this function unconditionally prints ANSI color codes.
/// It should only be set to true only if it is known that the terminal supports it.
pub fn pretty_print_assertion(assert: &AssertInfo<'_>, loc: Location<'_>, ansi_colors: bool) {
    let macro_name = assert.macro_name;
    if ansi_colors {
        print_pretty_header(loc);
        match &assert.assertion {
            Assertion::Bool(assert) => print_pretty_bool_assertion(macro_name, assert),
            Assertion::Binary(assert) => print_pretty_binary_assertion(macro_name, assert),
        }
        if let Some(msg) = &assert.message {
            print_pretty_message(msg);
        }
    } else {
        print_plain_header(loc);
        match &assert.assertion {
            Assertion::Bool(assert) => print_plain_bool_assertion(macro_name, assert),
            Assertion::Binary(assert) => print_plain_binary_assertion(macro_name, assert),
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

fn print_plain_bool_assertion(macro_name: &'static str, assert: &BoolAssertion) {
    eprint!(
        concat!(
            "Assertion:\n",
            "  {macro_name}!( {expr} )\n",
            "Expansion:\n",
            "  {macro_name}!( false )\n",
        ),
        macro_name = macro_name,
        expr = assert.expr,
    )
}

fn print_pretty_bool_assertion(macro_name: &'static str, assert: &BoolAssertion) {
    eprint!(
        concat!(
            "{bold}Assertion:{reset}\n",
            "  {magenta}{macro_name}!( {cyan}{expr} {magenta}){reset}\n",
            "{bold}Expansion:{reset}\n",
            "  {magenta}{macro_name}!( {cyan}false {magenta}){reset}\n",
        ),
        magenta = MAGENTA,
        cyan = CYAN,
        reset = RESET,
        bold = BOLD,
        macro_name = macro_name,
        expr = assert.expr,
    );
}

fn print_plain_binary_assertion(macro_name: &'static str, assert: &BinaryAssertion<'_>) {
    if is_specalized_macro(macro_name) {
        eprint!(
            concat!(
                "Assertion:\n",
                "  {macro_name}!( {left_expr}, {right_expr} )\n",
                "Expansion:\n",
                "  {macro_name}!( {left_val:?}, {right_val:?} )\n",
            ),
            macro_name = macro_name,
            left_expr = assert.left_expr,
            right_expr = assert.right_expr,
            left_val = assert.left_val,
            right_val = assert.right_val,
        );
    } else {
        eprint!(
            concat!(
                "Assertion:\n",
                "  {macro_name}!( {left_expr} {op} {right_expr} )\n",
                "Expansion:\n",
                "  {macro_name}!( {left_val:?} {op} {right_val:?} )\n",
            ),
            macro_name = macro_name,
            op = assert.op,
            left_expr = assert.left_expr,
            right_expr = assert.right_expr,
            left_val = assert.left_val,
            right_val = assert.right_val,
        );
    };
}

fn print_pretty_binary_assertion(macro_name: &'static str, assert: &BinaryAssertion<'_>) {
    if is_specalized_macro(macro_name) {
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
            macro_name = macro_name,
            left_expr = assert.left_expr,
            right_expr = assert.right_expr,
            left_val = assert.left_val,
            right_val = assert.right_val,
        );
    } else {
        eprint!(
            concat!(
                "{bold}Assertion:{reset}\n",
                "  {magenta}{macro_name}!( {cyan}{left_expr} {bold}{blue}{op}{reset} {yellow}{right_expr} {magenta}){reset}\n",
                "{bold}Expansion:{reset}\n",
                "  {magenta}{macro_name}!( {cyan}{left_val:?} {bold}{blue}{op}{reset} {yellow}{right_val:?} {magenta}){reset}\n",
            ),
            blue = BLUE,
            cyan = CYAN,
            magenta = MAGENTA,
            yellow = YELLOW,
            bold = BOLD,
            reset = RESET,
            macro_name = macro_name,
            op = assert.op,
            left_expr = assert.left_expr,
            right_expr = assert.right_expr,
            left_val = assert.left_val,
            right_val = assert.right_val,
        );
    };
}

fn print_plain_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("Message:\n  {}", message);
}

fn print_pretty_message(message: &std::fmt::Arguments<'_>) {
    eprintln!("{bold}Message:{reset}\n  {msg}", bold = BOLD, reset = RESET, msg = message,);
}

fn is_specalized_macro(macro_name: &str) -> bool {
    // Specialized macros already imply the operator in their name,
    // so we print them without repeating the operator.
    macro_name == "assert_eq" || macro_name == "assert_neq" || macro_name == "assert_matches"
}

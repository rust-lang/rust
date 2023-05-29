#![allow(unused)]
#![warn(clippy::impossible_double_const_comparisons)]
#![warn(clippy::ineffective_double_const_comparisons)]
#![allow(clippy::no_effect)]
#![allow(clippy::short_circuit_statement)]
#![allow(clippy::manual_range_contains)]

const STATUS_BAD_REQUEST: u16 = 400;
const STATUS_SERVER_ERROR: u16 = 500;

fn main() {
    let status_code = 500; // Value doesn't matter for the lint

    status_code >= 400 && status_code < 500; // Correct
    status_code <= 400 && status_code > 500;
    status_code > 500 && status_code < 400;
    status_code < 500 && status_code > 500;

    // More complex expressions
    status_code < { 400 } && status_code > { 500 };
    status_code < STATUS_BAD_REQUEST && status_code > STATUS_SERVER_ERROR;
    status_code <= u16::MIN + 1 && status_code > STATUS_SERVER_ERROR;
    status_code < STATUS_SERVER_ERROR && status_code > STATUS_SERVER_ERROR;

    // Yoda conditions
    500 <= status_code && 600 > status_code; // Correct
    500 <= status_code && status_code <= 600; // Correct
    500 >= status_code && 600 < status_code; // Incorrect
    500 >= status_code && status_code > 600; // Incorrect

    // Expressions where one of the sides has no effect
    status_code < 200 && status_code <= 299;
    status_code > 200 && status_code >= 299;

    status_code >= 500 && status_code > 500; // Useless left
    status_code > 500 && status_code >= 500; // Useless right
    status_code <= 500 && status_code < 500; // Useless left
    status_code < 500 && status_code <= 500; // Useless right

    // Other types
    let name = "Steve";
    name < "Jennifer" && name > "Shannon";

    let numbers = [1, 2];
    numbers < [3, 4] && numbers > [5, 6];

    let letter = 'a';
    letter < 'b' && letter > 'c';

    let area = 42.0;
    area < std::f32::consts::E && area > std::f32::consts::PI;
}

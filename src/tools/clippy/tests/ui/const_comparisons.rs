#![allow(unused)]
#![warn(clippy::impossible_comparisons)]
#![warn(clippy::redundant_comparisons)]
#![allow(clippy::no_effect)]
#![allow(clippy::short_circuit_statement)]
#![allow(clippy::manual_range_contains)]

const STATUS_BAD_REQUEST: u16 = 400;
const STATUS_SERVER_ERROR: u16 = 500;

struct Status {
    code: u16,
}

impl PartialEq<u16> for Status {
    fn eq(&self, other: &u16) -> bool {
        self.code == *other
    }
}

impl PartialOrd<u16> for Status {
    fn partial_cmp(&self, other: &u16) -> Option<std::cmp::Ordering> {
        self.code.partial_cmp(other)
    }
}

impl PartialEq<Status> for u16 {
    fn eq(&self, other: &Status) -> bool {
        *self == other.code
    }
}

impl PartialOrd<Status> for u16 {
    fn partial_cmp(&self, other: &Status) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.code)
    }
}

fn main() {
    let status_code = 500; // Value doesn't matter for the lint
    let status = Status { code: status_code };

    status_code >= 400 && status_code < 500; // Correct
    status_code <= 400 && status_code > 500;
    status_code > 500 && status_code < 400;
    status_code < 500 && status_code > 500;

    // More complex expressions
    status_code < { 400 } && status_code > { 500 };
    status_code < STATUS_BAD_REQUEST && status_code > STATUS_SERVER_ERROR;
    status_code <= u16::MIN + 1 && status_code > STATUS_SERVER_ERROR;
    status_code < STATUS_SERVER_ERROR && status_code > STATUS_SERVER_ERROR;

    // Comparing two different types, via the `impl PartialOrd<u16> for Status`
    status < { 400 } && status > { 500 };
    status < STATUS_BAD_REQUEST && status > STATUS_SERVER_ERROR;
    status <= u16::MIN + 1 && status > STATUS_SERVER_ERROR;
    status < STATUS_SERVER_ERROR && status > STATUS_SERVER_ERROR;

    // Yoda conditions
    500 <= status_code && 600 > status_code; // Correct
    500 <= status_code && status_code <= 600; // Correct
    500 >= status_code && 600 < status_code; // Incorrect
    500 >= status_code && status_code > 600; // Incorrect

    // Yoda conditions, comparing two different types
    500 <= status && 600 > status; // Correct
    500 <= status && status <= 600; // Correct
    500 >= status && 600 < status; // Incorrect
    500 >= status && status > 600; // Incorrect

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

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

    // Correct
    status_code >= 400 && status_code < 500;
    status_code <= 400 && status_code > 500;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `400` < `500`, the expression evaluates to false for any value of `st
    status_code > 500 && status_code < 400;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `500` > `400`, the expression evaluates to false for any value of `st
    status_code < 500 && status_code > 500;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: `status_code` cannot simultaneously be greater than and less than `500`

    // More complex expressions
    status_code < { 400 } && status_code > { 500 };
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `{ 400 }` < `{ 500 }`, the expression evaluates to false for any valu
    status_code < STATUS_BAD_REQUEST && status_code > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `STATUS_BAD_REQUEST` < `STATUS_SERVER_ERROR`, the expression evaluate
    status_code <= u16::MIN + 1 && status_code > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `u16::MIN + 1` < `STATUS_SERVER_ERROR`, the expression evaluates to f
    status_code < STATUS_SERVER_ERROR && status_code > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: `status_code` cannot simultaneously be greater than and less than `STATUS_S

    // Comparing two different types, via the `impl PartialOrd<u16> for Status`
    status < { 400 } && status > { 500 };
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `{ 400 }` < `{ 500 }`, the expression evaluates to false for any valu
    status < STATUS_BAD_REQUEST && status > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `STATUS_BAD_REQUEST` < `STATUS_SERVER_ERROR`, the expression evaluate
    status <= u16::MIN + 1 && status > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `u16::MIN + 1` < `STATUS_SERVER_ERROR`, the expression evaluates to f
    status < STATUS_SERVER_ERROR && status > STATUS_SERVER_ERROR;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: `status` cannot simultaneously be greater than and less than `STATUS_SERVER

    // Yoda conditions
    // Correct
    500 <= status_code && 600 > status_code;
    // Correct
    500 <= status_code && status_code <= 600;
    // Incorrect
    500 >= status_code && 600 < status_code;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `500` < `600`, the expression evaluates to false for any value of `st
    // Incorrect
    500 >= status_code && status_code > 600;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `500` < `600`, the expression evaluates to false for any value of `st

    // Yoda conditions, comparing two different types
    // Correct
    500 <= status && 600 > status;
    // Correct
    500 <= status && status <= 600;
    // Incorrect
    500 >= status && 600 < status;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `500` < `600`, the expression evaluates to false for any value of `st
    // Incorrect
    500 >= status && status > 600;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `500` < `600`, the expression evaluates to false for any value of `st

    // Expressions where one of the sides has no effect
    status_code < 200 && status_code <= 299;
    //~^ ERROR: right-hand side of `&&` operator has no effect
    status_code > 200 && status_code >= 299;
    //~^ ERROR: left-hand side of `&&` operator has no effect

    // Useless left
    status_code >= 500 && status_code > 500;
    //~^ ERROR: left-hand side of `&&` operator has no effect
    // Useless right
    status_code > 500 && status_code >= 500;
    //~^ ERROR: right-hand side of `&&` operator has no effect
    // Useless left
    status_code <= 500 && status_code < 500;
    //~^ ERROR: left-hand side of `&&` operator has no effect
    // Useless right
    status_code < 500 && status_code <= 500;
    //~^ ERROR: right-hand side of `&&` operator has no effect

    // Other types
    let name = "Steve";
    name < "Jennifer" && name > "Shannon";
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `"Jennifer"` < `"Shannon"`, the expression evaluates to false for any

    let numbers = [1, 2];
    numbers < [3, 4] && numbers > [5, 6];
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `[3, 4]` < `[5, 6]`, the expression evaluates to false for any value

    let letter = 'a';
    letter < 'b' && letter > 'c';
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `'b'` < `'c'`, the expression evaluates to false for any value of `le

    let area = 42.0;
    area < std::f32::consts::E && area > std::f32::consts::PI;
    //~^ ERROR: boolean expression will never evaluate to 'true'
    //~| NOTE: since `std::f32::consts::E` < `std::f32::consts::PI`, the expression evalua
}
